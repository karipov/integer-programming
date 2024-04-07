
def check_integer_solution(solution) -> bool:
    """
    Checks if our LP relaxation solution is a solution to the IP problem
    """
    threshold = 1e-5
    thresh_check = lambda x: abs(x - round(x)) > threshold
    return all(map(thresh_check, solution.as_df()['value']))