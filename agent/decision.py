"""
Agent Decision Module
"""

from .planner import Plan, StepStatus


class DecisionMaker:
    """
    Decide the next executable step.
    """

    def next_step(self, plan: Plan):
        """
        Return the next pending step.
        """

        for step in plan.steps:
            if step.status == StepStatus.PENDING:
                return step

        return None
