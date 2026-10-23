"""Dependencies the routes share.

The authenticated user (2.1.9: ``X-API-Key`` mapped to a ``user_id``) and
the compiled agent graph built at startup (2.1.4) are provided here, so a
route declares what it needs and tests can override it.
"""
