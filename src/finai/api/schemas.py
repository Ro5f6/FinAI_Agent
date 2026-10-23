"""Request and response shapes of the API (2.1.3).

``ChatRequest`` is what a client sends; ``StreamEvent`` is each server-sent
event of the answer stream. Neither carries a ``user_id``: the user is
always taken from authentication (2.1.9).
"""
