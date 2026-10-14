"""Regression checks for edits to the visible, generated benchmark values."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
import shutil

import check_site


class DisplayedResultsTest(TestCase):
    def test_manually_changed_table_and_metric_card_are_rejected(self):
        with TemporaryDirectory() as temporary:
            site = Path(temporary)
            (site / 'assets').mkdir()
            shutil.copy(check_site.SITE / 'assets/results.json', site / 'assets/results.json')
            original = (check_site.SITE / 'index.html').read_text()
            for before, after in (('<td class="ffhq-fid">75.9</td>', '<td class="ffhq-fid">70.0</td>'),
                                  ('1.52 <span>→</span> 1.31', '1.52 <span>→</span> 1.01')):
                self.assertIn(before, original)
                (site / 'index.html').write_text(original.replace(before, after, 1))
                with patch.object(check_site, 'SITE', site):
                    with self.assertRaisesRegex(AssertionError, 'differs from results.json'):
                        check_site.check_displayed_results()
                    check_site.check_displayed_results(write=True)
                    check_site.check_displayed_results()
                self.assertEqual((site / 'index.html').read_text(), original)

    def test_paper_coverage_columns_keep_caption_mapping(self):
        sources = check_site.expected_figure_sources()
        self.assertTrue(sources['ffhq-pca-nfe-2.jpg'].endswith('latent_pca_real_fit_5.jpg'))
        self.assertTrue(sources['ffhq-umap-nfe-10.jpg'].endswith('latent_umap_real_fit_50.jpg'))


if __name__ == '__main__':
    main()
