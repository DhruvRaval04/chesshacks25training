import requests

def get_chess_evaluation(fen_string):
    """
    Analyzes a chess position using the chess-api.com API.

    Args:
        fen_string (str): The FEN string of the chess position to analyze.

    Returns:
        float: The evaluation score in pawns, or None if an error occurs.
    """
    url = "https://chess-api.com/v1"
    payload = {"fen": fen_string}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get('eval')
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None