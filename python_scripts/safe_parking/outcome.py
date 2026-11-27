"""Keep solver termination, validation and acceptance as separate observations."""
SUCCESS = ('Solve_Succeeded', 'Solved_To_Acceptable_Level')


def solver_stop(status):
    return {'Maximum_Iterations_Exceeded':'ITERATION_LIMIT',
            'Maximum_CpuTime_Exceeded':'SOLVER_TIME_LIMIT',
            'Maximum_WallTime_Exceeded':'SOLVER_TIME_LIMIT',
            'Infeasible_Problem_Detected':'SOLVER_REPORTED_INFEASIBLE',
            'User_Requested_Stop':'USER_STOP'}.get(status, 'SOLVER_TERMINATED')


def outcome(solver_status, stop_reason, validation, **extra):
    accepted = bool(solver_status in SUCCESS and validation and validation['accepted'])
    failures = list(validation.get('failure_items', [])) if validation else []
    if solver_status not in SUCCESS:
        failures.append('solver_not_successful' if solver_status else 'no_solver_result')
    return dict(status='PASS_CONDITIONAL_NUMERICAL' if accepted else 'NOT_ACCEPTED',
                accepted=accepted, solver_status=solver_status, stop_reason=stop_reason,
                validation_status=validation['curve']['state'] if validation else 'NOT_RUN',
                validation=validation, failure_items=failures, **extra)
