## Directory Structure

```plaintext
.copilot/
├── feature-development-pattern.md          # Details the pattern for feature development
├── feat/                                   # Contains documentation for features
│   ├── lessons-learned.md                  # Captures key learnings from implementations
│   └── US12345-dynamic-message-group-id/   # Example feature directory
│       ├── intro.md                        # Feature introduction and context
│       ├── implementation-plan.md          # Detailed implementation milestones
│       └── implementation-progress.md      # Tracks progress of implementation
└── deps/                                   # Documentation on dependencies and interfaces
    ├── sample-payload.md                   # Sample payloads
    ├── architectural-info.md               # Architectural information
    └── dependency-docs.md                  # Dependency documentation
```

## Usage Guidelines

### 1. Create a New Feature Directory

To start a new feature, request Copilot to create a directory under `feat/` using the following prompt:

```plaintext
Please review the feature development pattern document. We are going to start working on a new feature: 
[feature-number]-[feature-name]
```

### 2. Document Feature Details

Populate the `intro.md` file with the following structure:

```markdown
# US12345 - Dynamic MessageGroupId

## Introduction
<!-- Describe the feature and its purpose here -->

## Goal
<!-- Define the specific goals and objectives of this feature -->

## Context
<!-- Provide any necessary context for understanding the feature -->

## Implementation Guidelines
<!-- List any specific guidelines or requirements for implementation -->
```

### 3. Plan Implementation

Collaborate with GitHub Copilot to create and refine the implementation plan:

```plaintext
I have filled out the intro doc. I am ready to start planning.
```

### 4. Review and Provide Feedback

After reviewing the implementation plan, provide feedback in the following format:

```plaintext
Feedback:
- Feedback comment 1
- Feedback comment 2
```

### 5. Start Implementation

Once satisfied with the implementation plan, proceed to the first milestone:

```plaintext
I have reviewed the implementation plan. It looks good. Let's start the first milestone.

If any changes are needed, provide feedback.
```

### 6. Proceed to next Milestone

After completing a milestone, move on to the next one:

```plaintext
The first milestone looks good. Let's move on to implementing milestone 2.
```

### 7. Track Progress

Update the `implementation-progress.md` file to track progress.

### 8. Document Learnings

Record key learnings in the `lessons-learned.md` file.
