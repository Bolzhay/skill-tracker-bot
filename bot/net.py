import httpx


def google_transport() -> httpx.AsyncHTTPTransport:
    """Some networks have a dead IPv6 route to Google: every connect to its IPv6
    addresses hangs until it times out, minutes in total. Binding to 0.0.0.0
    keeps calls to Google's APIs on IPv4."""
    return httpx.AsyncHTTPTransport(local_address="0.0.0.0")
