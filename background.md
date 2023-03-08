
# 代码结构

## Environment

### Jobs

#### Class JobDistribution

- init class
  - max new work vector
  - max job length
- method
  - normal dist
  - bi model dist

#### Class JobCollection

- init class
  - average job comming per slot
- attributes
  - average
  - id_start
  - enter time
  - distribution
  - now id
- method
  - get_job_collection

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
  - job vector
  - price
- method
  - show()
  - start()
  - finish()
  - to_list()
  - generate_job()

#### Class JobCollection

- init class
  - average
  - id start
  - enter time
  - duration
  - job distribution
- attribute
  - average
  - id start
  - enter time
  - duration
  - job distribution
- method
  - get_job_collection
  - get_job_collections

#### Class JobPreallocation

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

#### Machine

- attribute
  - id
  - number of resource
  - time horizon
  - reward
  - cost_vector
  - resource slot
  - available slot
  - running job
- method
  - allocate job
  - time proceed
  - show

### Users

### Agents
