"""Validate page links, figure provenance, and the manuscript-results data file.

Run with --manuscript-root PATH to also verify original bytes, active table
values and figure inclusions. The manuscript is not required in a public clone.
"""
from argparse import ArgumentParser
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json
import math
import re
from html import escape

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'docs'
DATASETS = ('moon', 'checkerboard_grid', 'letter_f', 'letter_m')
TRAJECTORY_METHODS = (
    'ground_truth', 'drift_flow_matching_steps_1', 'drift_flow_matching_steps_20',
    'flow_matching', 'mean_flow_steps_1', 'mean_flow_steps_20', 'drift',
)
GROUP_PANELS = {
    'source-target': ('01_source_target.jpg', None),
    'early': ('group_early_drift.jpg', [0.05, 0.30]),
    'wide': ('group_wide_drift.jpg', [0.20, 0.80]),
    'middle': ('group_middle_drift.jpg', [0.35, 0.65]),
    'late': ('group_late_drift.jpg', [0.70, 0.95]),
}
# Column captions in the active paper figure are authoritative; filenames retain
# their original experiment suffixes.
FFHQ_COVERAGE = ((1, 1), (2, 5), (5, 10), (10, 50))
RESULT_SOURCES = {
    'generation_table': 'tab/tab_combine.tex',
    'robotics_table': 'tab/tab_robotic.tex',
    'experiment_protocol': 'sec/05_experiment.tex',
}


def render_results(data):
    """Render all displayed benchmark numbers from the manuscript data file."""
    rows = data['mnist_ffhq']['rows']
    dfm = [row for row in rows if row['method'] == 'DFM']
    imagenet = data['imagenet']['rows']
    robot = data['robotics']
    toolhang = robot['series'][0]['success_rate']
    arrow = ' <span>→</span> '
    cards = [
        ('FFHQ · FID ↓', arrow.join(f'{row["ffhq_fid"]:.1f}' for row in dfm),
         ' → '.join(str(row['nfe']) for row in dfm) + ' NFE', 'Decoded-real reference protocol'),
        ('ImageNet 256 × 256 · FID ↓', arrow.join(f'{row["fid"]:.2f}' for row in (imagenet[0], imagenet[-1])),
         f'{imagenet[0]["nfe"]} → {imagenet[-1]["nfe"]} NFE', f'DFM-L/2 · {data["imagenet"]["parameters_millions"]}M parameters'),
        ('ToolHang · state success ↑', f'{toolhang[0]:.2f}' + arrow + f'{toolhang[-1]:.2f}',
         f'{robot["nfe"][0]} → {robot["nfe"][-1]} NFE', 'Average over the last 10 checkpoints'),
    ]
    rendered = {'cards': '<div class="metric-grid">\n' + '\n'.join(
        f'<article class="metric-card"><span class="card-kicker">{title}</span><p class="metric-values">{values}</p><p>{budget}</p><p class="small-note">{note}</p></article>'
        for title, values, budget, note in cards) + '\n</div>'}
    image_rows = []
    for row in rows:
        attrs = f' class="ours" data-nfe="{row["nfe"]}"' if row['method'] == 'DFM' else ''
        method = escape(row['method'].replace('Mean Flow', 'MeanFlow'))
        image_rows.append(f'<tr{attrs}><th scope="row">{method}</th><td>{row["nfe"]}</td>'
                          f'<td>{row["mnist_emd"]:.1f}</td><td>{row["mnist_accuracy_percent"]:.1f}%</td>'
                          f'<td class="ffhq-emd">{row["ffhq_emd"]:.1f}</td><td class="ffhq-fid">{row["ffhq_fid"]:.1f}</td></tr>')
    rendered['image-table'] = '<tbody id="image-results">\n' + '\n'.join(image_rows) + '\n</tbody>'
    rendered['imagenet-table'] = '<tbody>\n' + '\n'.join(
        f'<tr><th scope="row">{row["nfe"]}</th><td>{row["fid"]:.2f}</td><td>{row["inception_score"]:.1f}</td></tr>'
        for row in imagenet) + '\n</tbody>'
    rendered['robotics-table'] = '<tbody>\n' + '\n'.join(
        f'<tr><th scope="row">{nfe}</th>' + ''.join(f'<td>{series["success_rate"][i]:.2f}</td>' for series in robot['series']) + '</tr>'
        for i, nfe in enumerate(robot['nfe'])) + '\n</tbody>'
    first = dfm[0]
    rendered['ffhq-metric'] = f'<p class="budget-note" id="ffhq-metric" aria-live="polite">FFHQ · {first["nfe"]} NFE · FID {first["ffhq_fid"]:.1f} · EMD {first["ffhq_emd"]:.1f}</p>'
    return rendered


def check_displayed_results(write=False):
    path = SITE / 'index.html'
    html = path.read_text()
    for name, content in render_results(json.loads((SITE / 'assets/results.json').read_text())).items():
        start, end = f'<!-- results:{name} -->', f'<!-- /results:{name} -->'
        pattern = re.compile(re.escape(start) + r'.*?' + re.escape(end), re.S)
        assert len(pattern.findall(html)) == 1, f'Missing or duplicate results block: {name}'
        expected = f'{start}\n{content}\n{end}'
        if write:
            html = pattern.sub(lambda _: expected, html)
        else:
            assert pattern.search(html)[0] == expected, f'Displayed {name} differs from results.json; run python scripts/check_site.py --write-results'
    if write:
        path.write_text(html)


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate HTML id: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for name in ('src', 'href'):
            if name in attrs:
                self.references.append(attrs[name])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_tex(path):
    """Ignore TeX comments, retaining escaped percent signs in table values."""
    return '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0] for line in path.read_text().splitlines())


def number(cell):
    cell = re.sub(r'\\cellcolor\{[^}]*\}', '', cell)
    match = re.search(r'\d+(?:\.\d+)?', cell)
    assert match, f'Missing numeric table cell: {cell!r}'
    return float(match.group())


def expected_figure_sources():
    sources = {
        f'{dataset}-{method}.jpg': f'img/optimized/img_viz/2d_circular_uniform_to_{dataset}_{method}.jpg'
        for dataset in DATASETS for method in TRAJECTORY_METHODS
    }
    sources.update({
        f'ffhq-nfe-{nfe}.jpg': f'img/optimized/result_FFHQ/ffhq_fake_steps_grid_step_{nfe}.pdf'
        for nfe in (1, 2, 5, 10)
    })
    sources.update({
        f'group-{panel}.jpg': f'img/optimized/img_group_drift/{filename}'
        for panel, (filename, _) in GROUP_PANELS.items()
    })
    sources.update({
        f'ffhq-{projection}-nfe-{nfe}.jpg': f'img/optimized/img_FFHQ/latent_{projection}_real_fit_{suffix}.jpg'
        for projection in ('pca', 'umap') for nfe, suffix in FFHQ_COVERAGE
    })
    return sources


def check_references():
    parser = References()
    parser.feed((SITE / 'index.html').read_text())
    for reference in parser.references:
        if reference.startswith('#'):
            assert not reference[1:] or reference[1:] in parser.ids, reference
        elif not reference.startswith(('https:', 'http:', 'mailto:', 'data:')):
            assert (SITE / reference.split('#')[0]).is_file(), reference


def check_figures(manuscript):
    manifest = json.loads((SITE / 'assets/provenance.json').read_text())
    by_name = {entry['asset']: entry for entry in manifest}
    assert len(by_name) == len(manifest), 'Duplicate figure manifest entries'
    expected_sources = expected_figure_sources()
    assert set(by_name) == set(expected_sources), (
        f'Figure manifest mismatch: missing={set(expected_sources) - set(by_name)}, '
        f'unexpected={set(by_name) - set(expected_sources)}'
    )
    assert {path.name for path in (SITE / 'assets').glob('*.jpg')} == set(by_name), 'Unlisted or missing JPEG assets'
    for name, expected_source in expected_sources.items():
        entry = by_name[name]
        assert entry['source'] == expected_source, f'Incorrect source for {name}'
        asset = SITE / 'assets' / name
        assert asset.is_file(), asset
        assert sha256(asset) == entry.get('asset_sha256', entry['source_sha256']), f'Changed asset bytes: {asset}'
        if expected_source.endswith('.jpg'):
            assert sha256(asset) == entry['source_sha256'], f'Copied source image was modified: {asset}'
        if manuscript:
            source = manuscript / expected_source
            assert source.is_file(), source
            assert sha256(source) == entry['source_sha256'], f'Changed manuscript source: {source}'
    for panel, (_, pair) in GROUP_PANELS.items():
        assert by_name[f'group-{panel}.jpg']['time_pair'] == pair, f'Wrong time pair for {panel}'
    for projection in ('pca', 'umap'):
        for column, (nfe, _) in enumerate(FFHQ_COVERAGE, 1):
            entry = by_name[f'ffhq-{projection}-nfe-{nfe}.jpg']
            assert (entry['paper_column'], entry['nfe'], entry['projection']) == (column, nfe, projection.upper())
            assert entry['source_figure'] == 'img/img_FFHQ.tex'
            if manuscript:
                assert sha256(manuscript / entry['source_figure']) == entry['source_figure_sha256']
    if manuscript:
        trajectory_figure = active_tex(manuscript / 'img/img_viz.tex')
        group_figure = active_tex(manuscript / 'img/img_group_drift.tex')
        coverage_figure = active_tex(manuscript / 'img/img_FFHQ.tex')
        assert [int(nfe) for nfe in re.findall(r'NFE\s*=\s*(\d+)', coverage_figure)] == [nfe for nfe, _ in FFHQ_COVERAGE]
        coverage_sources = re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', coverage_figure)
        assert coverage_sources == [expected_sources[f'ffhq-{projection}-nfe-{nfe}.jpg'] for projection in ('pca', 'umap') for nfe, _ in FFHQ_COVERAGE]
        for name, source in expected_sources.items():
            if '/img_viz/' in source:
                assert source in trajectory_figure, f'Figure is not included in manuscript trajectory panel: {name}'
            elif '/img_group_drift/' in source:
                assert source in group_figure, f'Figure is not included in manuscript method panel: {name}'
        for _, pair in GROUP_PANELS.values():
            if pair:
                assert f'({pair[0]:.2f},{pair[1]:.2f})' in group_figure, pair
    return len(manifest)


def check_results(manuscript):
    data = json.loads((SITE / 'assets/results.json').read_text())
    assert data['schema_version'] == 1
    for name, path in RESULT_SOURCES.items():
        assert data['sources'][name]['path'] == path, name
        assert re.fullmatch(r'[0-9a-f]{64}', data['sources'][name]['sha256']), name
        if manuscript:
            assert sha256(manuscript / path) == data['sources'][name]['sha256'], f'Changed result source: {path}'
    assert data['mnist_ffhq']['source'] == data['imagenet']['source'] == RESULT_SOURCES['generation_table']
    assert data['robotics']['source'] == RESULT_SOURCES['robotics_table']
    for section in ('mnist_ffhq', 'imagenet', 'robotics'):
        assert data[section]['scope'] and data[section]['release_scope'] and data[section]['protocol'], section

    image_rows = data['mnist_ffhq']['rows']
    image_columns = ('mnist_emd', 'mnist_accuracy_percent', 'ffhq_emd', 'ffhq_fid')
    assert [(row['method'], row['nfe']) for row in image_rows] == [
        ('Flow Matching', 50), ('Mean Flow', 1), ('Mean Flow', 2), ('Mean Flow', 5),
        ('Drift Model', 1), ('DFM', 1), ('DFM', 2), ('DFM', 5),
    ], 'Incomplete or altered MNIST/FFHQ comparison rows'
    for row in image_rows:
        assert all(math.isfinite(row[key]) and row[key] >= 0 for key in image_columns)
        assert row['mnist_accuracy_percent'] <= 100
    assert [row['nfe'] for row in data['imagenet']['rows']] == [1, 2, 5, 10]
    assert all(math.isfinite(row[key]) and row[key] > 0 for row in data['imagenet']['rows'] for key in ('fid', 'inception_score'))
    robot = data['robotics']
    assert robot['nfe'] == [1, 2, 5, 10]
    assert [(series['task'], series['setting']) for series in robot['series']] == [('ToolHang', 'State'), ('PushT', 'Visual')]
    for series in robot['series']:
        assert len(series['success_rate']) == len(robot['nfe'])
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in series['success_rate'])

    if not manuscript:
        return
    protocol = active_tex(manuscript / RESULT_SOURCES['experiment_protocol'])
    assert r'\input{tab/tab_combine}' in protocol and r'\input{tab/tab_robotic}' in protocol
    assert r'\input{tab/tab_imagenet}' not in protocol, 'Unexpected active ImageNet table: review results provenance'
    combined = active_tex(manuscript / RESULT_SOURCES['generation_table'])
    left, right = combined.split(r'\end{tabular}', maxsplit=1)
    recovered_images = []
    for row in left.split(r'\\'):
        cells = row.split('&')
        match = re.search(r'(Flow Matching|Mean Flow|Drift Model|DFM)\$\^\{(\d+)\}\$', cells[0])
        if match:
            assert len(cells) == 5
            recovered_images.append(dict(method=match[1], nfe=int(match[2]), **dict(zip(image_columns, map(number, cells[1:])))))
    assert image_rows == recovered_images, 'MNIST/FFHQ values differ from active manuscript table'
    recovered_imagenet = []
    for row in right.split(r'\\'):
        cells = row.split('&')
        if re.search(r'DFM, L/2', cells[0]):
            assert len(cells) == 5
            assert int(number(cells[1])) == data['imagenet']['parameters_millions']
            recovered_imagenet.append(dict(nfe=int(number(cells[2])), fid=number(cells[3]), inception_score=number(cells[4])))
    assert data['imagenet']['rows'] == recovered_imagenet, 'ImageNet values differ from active manuscript table'
    robot_tex = active_tex(manuscript / RESULT_SOURCES['robotics_table'])
    assert [int(value) for value in re.findall(r'NFE:\s*(\d+)', robot_tex)] == [100, 1, *robot['nfe']]
    recovered_robotics = []
    task = None
    for row in robot_tex.split(r'\\'):
        cells = [cell.strip() for cell in row.split('&')]
        if len(cells) != 9:
            continue
        if cells[1] in ('Lift', 'Can', 'ToolHang', 'PushT', 'BlockPush', 'Kitchen'):
            task = cells[1]
        if (task, cells[2]) in [('ToolHang', 'State'), ('PushT', 'Visual')]:
            recovered_robotics.append(dict(task=task, setting=cells[2], success_rate=[number(cell) for cell in cells[5:]]))
    assert robot['series'] == recovered_robotics, 'Robotics values differ from active manuscript table'


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript-root', type=Path, help='Check original manuscript files and active table values as well as released artifacts.')
    parser.add_argument('--write-results', action='store_true', help='Regenerate the static HTML result blocks from results.json before checking.')
    args = parser.parse_args()
    manuscript = args.manuscript_root
    if manuscript is None:
        manuscript = next((parent for parent in ROOT.parents if (parent / 'sec/05_experiment.tex').is_file()), None)
    if manuscript:
        manuscript = manuscript.resolve()
        assert (manuscript / 'sec/05_experiment.tex').is_file(), f'Not a manuscript root: {manuscript}'
    check_displayed_results(write=args.write_results)
    check_references()
    count = check_figures(manuscript)
    check_results(manuscript)
    scope = 'original source bytes, figure labels and active table values' if manuscript else 'release hashes and result schema (original manuscript unavailable)'
    print(f'Validated page references, {count} archived figures, and results data; checked {scope}.')


if __name__ == '__main__':
    main()
