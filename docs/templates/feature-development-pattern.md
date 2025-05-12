# Feature Development Pattern
This document describes the pattern used by the software engineer and Copilot while working on a feature.

# Context
A software developer is going to be working in concert with Copilot to implement a feature in this software project.

## Documentation
- The base documentation folder for feature development is `.copilot/feat`.
- The dependencies can be under `.copilot/deps`.
- All documentation should be in markdown format.
- The documents will be used by Copilot to help guide and track implementation.
- All new documents should start blank.
- The `.copilot/feat/lessons-learned.md` document captures key lessons learned during the implementation of features. It is where we will store new information which is discovered during the course of implementation. It should be used by Copilot when making implementation choices. Lessons learned should only include new lessons.
- The `.copilot/deps/` directory contains documentation on the dependencies, integrations, and interfaces used in this project.
- Feature documentation is organized in a consistent manner under `.copilot/feat/US[ticket-number]-[feature-name]/` with standardized document naming:
  - `intro.md`: Feature introduction and context.
    - Create place holder sections like Introduction, Goal, Context, Implementation Guidelines
  - `implementation-plan.md`: Detailed implementation milestones.
    - Each milestone will include a section describing the expected outcome.
    - The implementation plan is high level.
    - The implementation plan should not contain code, technical specifications, or directory structures. Prose only.
    - Each implementation milestone should include only a single discrete testable change. Break complex implementations into multiple smaller milestones, where each milestone can be independently verified.
  - `implementation-progress.md`: Track progress of implementation.
    - Checklists make the progress report easy to understand.

# Software Developer and Copilot Working Together
The feature will be implemented by Copilot with direction and feedback given by the software engineer. Focus on understanding the work to be done before getting to the code.

## Software Engineer's Role
The software engineer plays an important role in the implementation pattern. The software engineer:
- Provides feedback on changes made by Copilot.
- Validates milestone completion.
- Provides context.
- Makes git commits.
- Gets documentation for Copilot.
- Decides when to continue to the next activity.

## Copilot's Role
Copilot manages the code. This pattern is single writer and Copilot is the writer. Copilot:
- Writes the code.
- Keeps documents up to date.
- Asks the software engineer for input.

# General Pattern
- Copilot creates the docs for the new feature.
- The software engineer writes the intro file.
- Copilot checks its understanding of the intro file with the software engineer until consensus is reached.
- Copilot creates an implementation plan document.
- The software engineer reviews the implementation plan and provides feedback.
- The software engineer and Copilot work together to refine the implementation plan until the software engineer is satisfied with the plan.
- **Copilot waits for explicit approval from the software engineer before starting the implementation of any milestone.**
- Copilot begins implementation work, starting with the first milestone.
- Copilot runs builds and tests to verify the correctness of its changes.
- After Copilot finishes implementation, the software engineer checks the code and application.
- The software engineer provides feedback to Copilot.
- Copilot makes updates to the implementation based on the software engineer's feedback.
- The software engineer and Copilot continue the milestone refinement cycle until the software engineer is satisfied.
- The milestone is considered complete and Copilot updates the implementation progress and lessons learned docs.
- At the software engineer's direction, Copilot begins work on the next milestone.
- This pattern continues until all the milestones are complete.

## Post Feature Completion
- Update project-level docs such as the README file.

## Directives
- Do not begin implementing a milestone until the software developer agrees.
- **Copilot must wait for explicit approval from the software engineer before starting the implementation of any milestone.**


## Review and Approval Process
- **Feedback Loop**: After Copilot generates the implementation plan, the software engineer should review it thoroughly and provide detailed feedback. Copilot should not proceed to implementation until the software engineer explicitly approves the plan.
- **Approval Confirmation**: Copilot should ask for explicit confirmation from the software engineer before starting the implementation of any milestone. This ensures that the software engineer has had sufficient time to review and approve the plan.

## Documentation Updates
- **Documentation Review**: Before starting the implementation, Copilot should ensure that all relevant documentation is up-to-date and has been reviewed by the software engineer.
- **Progress Tracking**: Copilot should regularly update the `implementation-progress.md` document with detailed progress reports, including any changes made based on feedback.

## Implementation Phase
- **Milestone Verification**: After completing each milestone, Copilot should run builds and tests to verify the correctness of the changes. The software engineer should then review the code and application to ensure it meets the requirements.
- **Feedback Integration**: Copilot should integrate feedback from the software engineer promptly and accurately. The milestone is only considered complete when the software engineer is satisfied with the implementation.

## Communication and Collaboration
- **Regular Check-ins**: Copilot should schedule regular check-ins with the software engineer to discuss progress, challenges, and next steps. This ensures continuous alignment and collaboration.
- **Clear Communication**: Copilot should communicate clearly and promptly with the software engineer, especially when seeking feedback or approval.

## Best Practices
- **Incremental Changes**: Break down complex implementations into smaller, manageable milestones. Each milestone should be independently testable and verifiable.
- **Proactive Documentation**: Keep documentation proactive and detailed. This helps in tracking progress and understanding the context of changes.
- **Quality Assurance**: Prioritize quality assurance by running thorough tests and reviews at each milestone. This helps in catching issues early and ensures a robust implementation.
- **Feedback and Plan Updates**: Ensure that feedback from the software engineer is promptly integrated into the implementation plan documentation. This helps in maintaining an accurate and up-to-date plan.
- **Thorough Review**: Allow sufficient time for the software engineer to review the implementation plan and provide feedback before proceeding to implementation. This ensures that the plan is well-understood and agreed upon.
- **Continuous Improvement**: Regularly update the `lessons-learned.md` document with new insights and improvements discovered during the implementation process. This helps in refining future implementations.
