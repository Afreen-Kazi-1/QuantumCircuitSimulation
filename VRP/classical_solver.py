# classical_solver.py
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_vrp_classical(distance_matrix, num_vehicles):
    n = len(distance_matrix)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(a, b):
        return distance_matrix[manager.IndexToNode(a)][manager.IndexToNode(b)]

    transit_index = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)

    if solution:
        total_cost = solution.ObjectiveValue()
        return total_cost
    else:
        return None
