   try:
        with request.urlopen(req, timeout=10) as response:
            return 200 <= response.status < 300
    except (error.URLError, error.HTTPError):
        return False
