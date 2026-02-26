"""Annotation helpers split from Corpus."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .corpus_state import _mark_processing_stage


class AnnotatePhaseMixin:
    def annotate(self, annotation_type: str = "text", fully_annotate: bool = True) -> None:
        """
        Annotate extracted sections with classification information.

        Args:
            annotation_type: Type of annotation to use: 'text' or 'chapter'
                           - 'text': Use text-based annotation with section titles (default)
                           - 'chapter': Use chapter-based annotation with chapter numbers
            fully_annotate: Whether to perform full annotation of sections (default: True)
        """
        self.logger.info("Running section classification...")

        # Check if input parquet file exists
        if not self.sections_parquet.exists():
            self.logger.error("Sections file not found: %s. Please run section() first.", self.sections_parquet)
            return

        # Check if section classifier model exists
        model_exists = self.section_classifier_model_path.exists()
        if not model_exists:
            self.logger.warning("Model file not found at %s. To train a new model, run GlossSectionClassifier.train_from_csv()", self.section_classifier_model_path)

        # If no trained model, skip annotation with a clear message
        if not model_exists:
            self.logger.warning(
                "No section-classifier model found at %s. "
                "If you are running from a git checkout (not the pip package), make sure the "
                "'models/section_classifier.joblib' file is present or pass "
                "section_classifier_model_path explicitly. Skipping annotation.",
                self.section_classifier_model_path
            )
            return

        model_path = str(self.section_classifier_model_path)
        # Classify sections and save output to 'classified_sections.parquet'
        self.classifier.classify_sections(
            input_parquet=str(self.sections_parquet),
            output_parquet=str(self.classified_parquet),
            model_path=model_path,
            n_cpus=4,
            column_name='title'
        )


        # Perform full annotation if requested
        if fully_annotate:
            self.logger.info("Performing full annotation...")

            # If we're using auto annotation and have document types and annotation mappings available
            if annotation_type == "auto" and self.filename_to_doctype and self.annotation_mapping:
                # Create a mapping from filename to annotation type based on document types
                filename_to_annotation = {}
                for filename, doc_type in self.filename_to_doctype.items():
                    # Look up the annotation method for this document type in our mapping
                    # Default to 'text' if no mapping exists
                    filename_to_annotation[filename] = self.annotation_mapping.get(doc_type, 'text')

                self.logger.info("Using document-type specific annotation based on metadata")

                # Read the classified parquet file
                df = pd.read_parquet(str(self.classified_parquet))

                # Group by filename and process each document according to its annotation type
                updated_groups = []

                for filename, group in df.groupby('filename'):
                    # Determine annotation type for this file
                    doc_annotation = filename_to_annotation.get(filename, 'text')

                    # Process according to annotation type
                    if doc_annotation == 'chapter':
                        self.logger.debug("Processing %s as chapter", filename)
                        updated_group = self.classifier.fully_annotate_chapter_group(group)
                    else:
                        self.logger.debug("Processing %s as text", filename)
                        updated_group = self.classifier.fully_annotate_text_group(group)

                    if updated_group is not None:
                        updated_groups.append(updated_group)

                # Concatenate and save results
                if updated_groups:
                    df_updated = pd.concat(updated_groups)
                    df_updated.to_parquet(str(self.fully_annotated_parquet), index=False)
                else:
                    self.logger.warning("No valid document groups to process. Output file not created.")
            else:
                # Use the standard fully_annotate method with the specified annotation type
                self.classifier.fully_annotate(
                    input_parquet=str(self.classified_parquet),
                    output_parquet=str(self.fully_annotated_parquet),
                    document_types=self.filename_to_doctype if self.filename_to_doctype else None,
                    annotation_type=annotation_type
                )

            # Use the fully annotated output for adding document types
            self._add_document_types(self.fully_annotated_parquet)
            self._update_processing_stage_in_parquet(self.fully_annotated_parquet, "annotate")
        else:
            # Add document types to the classified output
            self._add_document_types(self.classified_parquet)
            self._update_processing_stage_in_parquet(self.classified_parquet, "annotate")

    def _add_document_types(self, parquet_file: Path) -> None:
        """
        Add document_type information to the classified sections.

        Args:
            parquet_file: Path to the Parquet file to update
        """
        if not self.filename_to_doctype:
            self.logger.warning("No document type information available. Skipping document type addition.")
            return

        if parquet_file.exists():
            try:
                # Read the parquet file
                df = pd.read_parquet(parquet_file)

                # Add document_type based on filename
                df['document_type'] = df['filename'].map(self.filename_to_doctype)

                # Check for missing document types
                missing_count = df['document_type'].isna().sum()
                if missing_count > 0:
                    self.logger.warning("%d sections (%.2f%%) have no document type!", missing_count, missing_count / len(df) * 100)
                    missing_filenames = df[df['document_type'].isna()]['filename'].unique()[:5]
                    self.logger.warning("Sample filenames with missing document types: %s", missing_filenames)

                    # Check if the issue might be due to .md extension
                    if any('.md' in str(f) for f in self.filename_to_doctype.keys()):
                        self.logger.warning("Possible cause: Metadata filenames contain .md extension but sections filenames don't")
                    elif any('.md' in str(f) for f in df['filename'].unique()[:100]):
                        self.logger.warning("Possible cause: Sections filenames contain .md extension but metadata filenames don't")

                # Save the updated file
                df.to_parquet(parquet_file, index=False)
                self.logger.info("Added document types to %s", parquet_file)
            except Exception as e:
                self.logger.error("Error adding document types: %s", e)
        else:
            self.logger.warning("File not found: %s", parquet_file)

    def _update_processing_stage_in_parquet(self, parquet_path: Path, stage: str) -> None:
        """Append *stage* to the ``processing_stage`` column of *parquet_path* and save in-place."""
        try:
            df = pd.read_parquet(parquet_path)
            if "processing_stage" in df.columns:
                df["processing_stage"] = df["processing_stage"].apply(
                    lambda x: _mark_processing_stage(str(x) if pd.notna(x) else "", stage)
                )
            else:
                df["processing_stage"] = stage
            df.to_parquet(parquet_path, index=False)
            self.logger.info("Updated processing_stage to include '%s' stage", stage)
        except Exception as e:
            self.logger.warning("Failed to update processing_stage in %s: %s", parquet_path, e)
