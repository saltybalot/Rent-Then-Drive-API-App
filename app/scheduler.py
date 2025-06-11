from ortools.sat.python import cp_model
import math
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_driver():
    drivers = [
        {"id": 0, "status": "Available", "location": (0, 0), "time_slots": [9, 11]},
        {"id": 1, "status": "On Trip", "location": (5, 5), "time_slots": [10, 11]},
        {"id": 2, "status": "Available", "location": (2, 3), "time_slots": [9, 10]},
    ]

    requests = [
        {"id": 0, "location": (1, 1), "time_slot": 9},
        {"id": 1, "location": (4, 4), "time_slot": 10},
    ]

    vehicle = {"id": 0, "driver_id": 0, "type": "Car"}
    owner_id = vehicle["driver_id"]  # Assume driver with id 0 is the owner of the requests

    model = cp_model.CpModel()

    num_drivers = len(drivers)
    num_requests = len(requests)
    # Binary assignment variables: assign[driver][request] = 1 if assigned
    assign = {}
    for d in range(num_drivers):
        for r in range(num_requests):
            assign[(d, r)] = model.NewBoolVar(f'assign_d{d}_r{r}')

    # Constraint: Each request is assigned to exactly one driver, with owner priority
    for r, request in enumerate(requests):
        owner = drivers[owner_id]
        # Check if owner is available and in the right time slot
        if owner["status"] == "Available" and request["time_slot"] in owner["time_slots"]:
            # Force owner to take this request
            model.Add(assign[owner_id, r] == 1)
            for d in range(num_drivers):
                if d != owner_id:
                    model.Add(assign[d, r] == 0)
        else:
            # If owner can't take it, assign to any available driver as before
            model.AddExactlyOne(assign[d, r] for d in range(num_drivers))


    # Constraint: Only available drivers, in matching time slots, can be assigned
    for d, driver in enumerate(drivers):
        for r, request in enumerate(requests):
            if driver["status"] != "Available" or request["time_slot"] not in driver["time_slots"]:
                model.Add(assign[d, r] == 0)

    # Optional: minimize total distance
    def euclidean(loc1, loc2):
        return math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])

    total_distance = model.NewIntVar(0, 10000, 'total_distance')
    distance_terms = []
    for d in range(num_drivers):
        for r in range(num_requests):
            dist = int(euclidean(drivers[d]["location"], requests[r]["location"]) * 100)  # scale to int
            distance_terms.append(assign[d, r] * dist)

    model.Add(total_distance == sum(distance_terms))
    model.Minimize(total_distance)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Assignment:")
        for d in range(num_drivers):
            for r in range(num_requests):
                if solver.Value(assign[d, r]):
                    print(f"  Driver {d} assigned to Request {r}")
    else:
        print("No feasible solution found.")
        return {"drivers": drivers, "requests": requests}
