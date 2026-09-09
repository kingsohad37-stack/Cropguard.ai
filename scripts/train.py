"""Real MobileNetV2 transfer-learning training on the generated PlantVillage split."""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG=(224,224); AUTOTUNE=tf.data.AUTOTUNE

def ds_from_csv(csv_path, labels, batch, shuffle, seed, max_per_class=0):
    df=pd.read_csv(csv_path); idx={x:i for i,x in enumerate(labels)}
    if max_per_class:
        # Deterministic, class-balanced subset for a genuinely trained quick demo.
        df = (df.groupby("label", group_keys=False)
                .apply(lambda group: group.sample(n=min(len(group), max_per_class), random_state=seed))
                .reset_index(drop=True))
    paths=df.path.values; ys=df.label.map(idx).values
    d=tf.data.Dataset.from_tensor_slices((paths,ys))
    if shuffle: d=d.shuffle(min(len(df),10000),seed=seed,reshuffle_each_iteration=True)
    def load(p,y):
        b=tf.io.read_file(p); x=tf.io.decode_image(b,channels=3,expand_animations=False)
        x=tf.image.resize(x,IMG); x=tf.cast(x,tf.float32); return x,y
    return d.map(load,num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)

def main():
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--fine-tune-epochs',type=int,default=4); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--max-per-class',type=int,default=0,help='Use a real balanced subset per split for a faster demo; 0 uses every image.'); args=ap.parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed); random.seed(args.seed); tf.keras.utils.set_random_seed(args.seed)
    labels=json.load(open('models/labels.json')); n=len(labels)
    train=ds_from_csv('data/splits/train.csv',labels,args.batch_size,True,args.seed,args.max_per_class); val=ds_from_csv('data/splits/val.csv',labels,args.batch_size,False,args.seed,args.max_per_class); test=ds_from_csv('data/splits/test.csv',labels,args.batch_size,False,args.seed,args.max_per_class)
    aug=keras.Sequential([layers.RandomFlip('horizontal'),layers.RandomRotation(.08),layers.RandomZoom(.12),layers.RandomContrast(.1)],name='augmentation')
    base=keras.applications.MobileNetV2(input_shape=(*IMG,3),include_top=False,weights='imagenet')
    base.trainable=False
    inp=keras.Input((*IMG,3)); x=aug(inp); x=keras.applications.mobilenet_v2.preprocess_input(x); x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.25)(x); out=layers.Dense(n,activation='softmax',name='predictions')(x)
    model=keras.Model(inp,out)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    Path('models').mkdir(exist_ok=True)
    ckpt='models/plantvillage_best.keras'
    cb=[keras.callbacks.ModelCheckpoint(ckpt,monitor='val_accuracy',save_best_only=True,mode='max'), keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3,restore_best_weights=True), keras.callbacks.CSVLogger('models/training_log.csv',append=False)]
    print('Stage 1: frozen ImageNet backbone')
    h1=model.fit(train,validation_data=val,epochs=args.epochs,callbacks=cb)
    # Fine tune upper MobileNetV2 layers.
    base.trainable=True
    for layer in base.layers[:-35]: layer.trainable=False
    model.compile(optimizer=keras.optimizers.Adam(args.lr*0.05),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print('Stage 2: fine-tuning upper MobileNetV2 layers')
    model.fit(train,validation_data=val,initial_epoch=args.epochs,epochs=args.epochs+args.fine_tune_epochs,callbacks=cb)
    best=keras.models.load_model(ckpt)
    test_loss,test_acc=best.evaluate(test,verbose=1)
    metrics={'test_loss':float(test_loss),'test_accuracy':float(test_acc),'best_checkpoint':ckpt,'classes':n,'epochs_stage1':args.epochs,'epochs_finetune':args.fine_tune_epochs,'batch_size':args.batch_size,'learning_rate':args.lr,'seed':args.seed,'max_per_class':args.max_per_class,'evaluated_on':'data/splits/test.csv'}
    Path('models/metrics.json').write_text(json.dumps(metrics,indent=2), encoding='utf-8')
    print('REAL TEST METRICS:',json.dumps(metrics,indent=2))
    # Keras format is the authoritative checkpoint. H5 is only a legacy
    # compatibility export, so never let an H5 serializer limitation discard
    # a valid real .keras model or its measured metrics.
    try:
        best.save('models/plantvillage_best.h5', include_optimizer=False)
        print('Saved compatibility H5 model to models/plantvillage_best.h5')
    except Exception as exc:
        h5_path = Path('models/plantvillage_best.h5')
        if h5_path.exists():
            h5_path.unlink()
        print(f'H5 compatibility export skipped: {exc}')
if __name__=='__main__': main()
