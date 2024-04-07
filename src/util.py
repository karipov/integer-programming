
def check_integer_solution(solution, num_tests) -> bool:
    """
    Checks if our LP relaxation solution is a solution to the IP problem
    """
    threshold = 1e-5
    thresh_check = lambda x: abs(x - round(x)) < threshold
    all_values = [solution.get_value(f"use_{i}") for i in range(num_tests)]
    return all(map(thresh_check, all_values))


