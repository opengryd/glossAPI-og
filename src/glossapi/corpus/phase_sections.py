"""Section extraction helpers split from Corpus."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd

from .corpus_state import _mark_processing_stage


class SectionPhaseMixin:
    def section(self) -> None:
        """
        Extract sections from markdown files and save to Parquet format.

        Uses files marked with 'good' extraction quality (if available) or all markdown files.
        """
        self.logger.info("Extracting sections from markdown files...")

        # Create output directory
        os.makedirs(self.sections_dir, exist_ok=True)

        # Determine which markdown files to section
        # Priority 1: self.good_files collected from clean()
        # Priority 2: legacy parquet 'extraction' column logic (for backward compatibility)
        # ------------------------------------------------------------------

        good_filenames: List[str] = []

        if getattr(self, "good_files", None):
            good_filenames = self.good_files
            self.logger.info("Using %d good filenames from clean()", len(good_filenames))
        else:
            # Fallback path: derive good filenames from parquet metadata
            self.logger.info("No good_files from clean(); using parquet filter/ocr_success if available")
            from glossapi.parquet_schema import ParquetSchema
            parquet_schema = ParquetSchema({'url_column': self.downloader_config.get('url_column', 'url')})
            parquet_path = self._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)

            if parquet_path is not None and parquet_path.exists():
                try:
                    df_meta = pd.read_parquet(parquet_path)
                    mask = pd.Series(False, index=df_meta.index)
                    if 'filter' in df_meta.columns:
                        mask = mask | (df_meta['filter'] == 'ok')
                    if 'ocr_success' in df_meta.columns:
                        mask = mask | (df_meta['ocr_success'].fillna(False))
                    good_rows = df_meta[mask]
                    # Legacy fallback: if nothing selected yet, try 'extraction' == 'good'
                    if good_rows.empty and 'extraction' in df_meta.columns:
                        legacy_rows = df_meta[df_meta['extraction'] == 'good']
                        if not legacy_rows.empty:
                            good_rows = legacy_rows
                    if not good_rows.empty and 'filename' in good_rows.columns:
                        good_filenames = [os.path.splitext(fn)[0] for fn in good_rows['filename'].astype(str).tolist() if fn]
                        self.logger.info("Selected %d files via metadata from %s", len(good_filenames), parquet_path)
                        # Update processing_stage for selected rows
                        try:
                            if 'processing_stage' not in df_meta.columns:
                                df_meta['processing_stage'] = pd.NA
                            sel_idx = good_rows.index
                            df_meta.loc[sel_idx, 'processing_stage'] = df_meta.loc[sel_idx, 'processing_stage'].apply(
                                lambda x: _mark_processing_stage(str(x) if pd.notna(x) else "", "section")
                            )
                            # Write back in-place
                            self._cache_metadata_parquet(parquet_path)
                            parquet_schema.write_metadata_parquet(df_meta, parquet_path)
                        except Exception as e:
                            self.logger.warning("Failed to update processing_stage in %s: %s", parquet_path, e)
                except Exception as e:
                    self.logger.warning("Error reading parquet file %s: %s", parquet_path, e)
            else:
                self.logger.info("No metadata parquet found for section selection; will fall back to all markdown files")

        self.logger.info("Found %d good quality files for sectioning", len(good_filenames))
        if good_filenames:
            self.logger.debug("Good filenames: %s", good_filenames)

        # Prefer cleaned markdown if available; otherwise fall back to raw markdown.
        markdown_root = self.cleaned_markdown_dir if self.cleaned_markdown_dir.exists() else self.markdown_dir

        if not good_filenames:
            self.logger.warning("No files marked as 'good' – falling back to processing all extracted markdown files.")
            good_filenames = [
                os.path.splitext(p.name)[0]
                for p in Path(markdown_root).glob("*.md")
            ]
            if not good_filenames:
                self.logger.warning(
                    "No markdown files found to section. Extraction might have failed. Skipping section phase."
                )
                return

        # Extract sections - pass list of good filenames to the sectioner
        # We will pass the original markdown directory and the list of good filenames 
        # rather than creating a separate directory
        self.sectioner.to_parquet(
            input_dir=str(markdown_root),  # Prefer cleaned markdown when present
            output_dir=str(self.sections_dir),
            filenames_to_process=good_filenames,
        )

        self.logger.info("Finished sectioning %d good quality files", len(good_filenames))
        self.logger.info("Section extraction complete. Parquet file saved to %s", self.sections_parquet)
