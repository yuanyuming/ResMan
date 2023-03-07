
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
  - 
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
- method
  - show()
  - start()
  - finish()
  - generate_job()

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
  - number of resource
  - time horizon
  - resource slot
  - available slot
  - running job
- method
  - allocate job
  - time proceed
  - show
### Users

### Agents
