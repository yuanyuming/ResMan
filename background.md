
# 代码结构

## Environment

### Jobs

#### Class Job

- init class
  - job id
  - resource vector
  - job len
  - enter time
- attribute
  - job id
  - resource vector
  - job len
  - enter time
  - start time
  - finish time
- method
  - show()
  - start()
  - finish()
  - generate_job()

#### Class JobDistribution

#### Class JobSlot

- init class
  - number of new jobs
- attribute
  - slot
- method
  - show()

#### Class JobBacklog

- init class
  - backlog size
- attribute
  - current size
  - backlog
- method
  - add_backlog()
  - show()

#### Class JobRecord

- attribute
  - record

### Servers

### Users

### Agents
