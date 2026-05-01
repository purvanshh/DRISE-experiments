Adds async document processing support with per-user result caching to improve throughput for concurrent users. Introduces a new endpoint for batch document uploads, a background processing queue using threading, and a caching layer that stores extracted text results keyed by user ID and filename.

TEST FIXTURE: Intentional flaws for PRGuard validation. Do not merge.
