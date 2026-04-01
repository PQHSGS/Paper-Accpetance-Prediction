class ScienceParse:
    """Entity for parsed PDF metadata; computation is handled outside the entity."""

    def __init__(
        self,
        title,
        abstract,
        sections,
        reference_titles,
        reference_venues,
        reference_years,
        reference_mention_contexts,
        reference_num_mentions,
        authors=None,
        emails=None,
        other_keys=None,
    ):
        self.title = title
        self.abstract = abstract
        self.sections = sections
        self.reference_titles = reference_titles
        self.reference_venues = reference_venues
        self.reference_years = reference_years
        self.reference_mention_contexts = reference_mention_contexts
        self.reference_num_mentions = reference_num_mentions
        self.authors = authors
        self.emails = emails
        self.other_keys = other_keys

        # Caches used by feature utility functions.
        self._cached_content = None
        self._cached_content_words = None
