create table if not exists public.scans (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  filename text not null,
  label text not null,
  confidence double precision not null check (confidence >= 0 and confidence <= 1),
  severity_score double precision not null check (severity_score >= 0 and severity_score <= 100),
  heatmap_coverage_percent double precision not null check (heatmap_coverage_percent >= 0 and heatmap_coverage_percent <= 100),
  image_sha256 text not null check (char_length(image_sha256) = 64)
);

create index if not exists scans_created_at_idx on public.scans (created_at desc);
create index if not exists scans_image_sha256_idx on public.scans (image_sha256);

alter table public.scans enable row level security;
-- There are intentionally no anon/authenticated policies: only the Render API,
-- using its private service-role secret, writes and reads scan history.
revoke all on public.scans from anon, authenticated;
