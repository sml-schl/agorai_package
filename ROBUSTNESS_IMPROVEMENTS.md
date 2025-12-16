# AgorAI Package - Robustness Improvements

## 🔴 Critical (Must Fix)

### 1. **Timeout Handling in Retry Loop**
**File:** `synthesis/core.py:154-168`
```python
# Problem: Infinite hang if LLM never responds
# Fix: Add timeout parameter
for attempt in range(self.max_retries + 1):
    try:
        result = asyncio.wait_for(
            self.generate(prompt),
            timeout=30.0  # Add timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout on attempt {attempt+1}")
        continue
```

### 2. **Empty Agent Outputs**
**File:** `synthesis/core.py:260-293`
```python
# Problem: IndexError if all agents fail
# Fix: Add validation
if not agent_outputs:
    raise ValueError("All agents failed to generate responses")
if not candidates or winner_idx is None:
    raise ValueError("No valid candidates found")
```

### 3. **Unvalidated API Inputs**
**File:** `backend/api/routes.py:68,721`
```python
# Problem: DoS via unlimited limit/max_items
# Fix: Add bounds checking
async def list_benchmarks(limit: int = 100, offset: int = 0):
    if not 1 <= limit <= 10000:
        raise HTTPException(400, "limit must be 1-10000")
    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")
```

### 4. **Agent Config Validation Missing**
**File:** `synthesis/core.py:46-64`
```python
# Problem: Invalid configs crash later
# Fix: Validate in __init__
if not provider or not model:
    raise ValueError("provider and model required")
if not 0.0 <= temperature <= 2.0:
    raise ValueError(f"temperature must be [0, 2], got {temperature}")
```

---

## 🟡 High Priority

### 5. **Add Comprehensive Logging**
**Files:** `synthesis/core.py`, `aggregate/core.py`
```python
import logging
logger = logging.getLogger(__name__)

# In generate_structured:
logger.info(f"Agent {self.name} generating (attempt {attempt+1})")
logger.warning(f"Validation failed: {validation.error_message}")

# In aggregate:
logger.debug(f"Aggregating {n_agents} agents, {n_candidates} candidates")
if maxv - minv <= 1e-12:
    logger.warning(f"Agent has constant utilities (all={u[0]})")
```

### 6. **Graceful Aggregation Fallback**
**File:** `aggregate/core.py:173-189`
```python
# Problem: Unknown method kills entire synthesis
# Fix: Fallback to majority
try:
    aggregator_func = AGGREGATOR_REGISTRY[method]
    result = aggregator_func(...)
except KeyError:
    logger.warning(f"Method {method} not found, using majority")
    result = AGGREGATOR_REGISTRY["majority"](...)
except Exception as e:
    logger.error(f"Aggregation failed: {e}", exc_info=True)
    raise
```

### 7. **Options Validation**
**File:** `synthesis/core.py:144-145`
```python
# Problem: Duplicates/empty options not rejected
# Fix: Add checks
if len(options) < 2:
    raise ValueError("Need at least 2 options")
if len(set(options)) != len(options):
    raise ValueError("Options must be unique")
if any(not opt.strip() for opt in options):
    raise ValueError("Options cannot be empty")
```

### 8. **Zero Division Protection**
**File:** `synthesis/core.py:289-293`
```python
# Problem: total_score can be 0
# Fix: Add guard
if scores:
    total_score = sum(scores)
    confidence = max(scores) / total_score if total_score > 0 else 0.0
else:
    confidence = 0.0
```

---

## 🟢 Medium Priority

### 9. **Performance: Tokenization Caching**
**File:** `synthesis/core.py:456-461`
```python
# Problem: Regex re-run per candidate
# Fix: Tokenize once
text_tokens = set(re.findall(r'\b\w+\b', text.lower()))
for candidate in candidates:
    cand_tokens = set(re.findall(r'\b\w+\b', candidate.lower()))
    overlap = len(text_tokens & cand_tokens)
    u.append(float(overlap))
```

### 10. **CSV Streaming**
**File:** `backend/api/routes.py:309-507`
```python
# Problem: Large CSVs load into memory
# Fix: Stream response
async def generate_csv():
    writer = csv.writer(output)
    for result in evaluation.results:
        yield writer.writerow([...])

return StreamingResponse(generate_csv(), media_type="text/csv")
```

### 11. **Thread-Safe Service Access**
**File:** `backend/api/routes.py:40-42`
```python
from threading import Lock

services_lock = Lock()
benchmark_service = None

def get_benchmark_service():
    global benchmark_service
    with services_lock:
        if benchmark_service is None:
            benchmark_service = BenchmarkService()
        return benchmark_service
```

### 12. **Rate Limiting**
**File:** `backend/api/routes.py`
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/inference/structured")
@limiter.limit("10/minute")  # Max 10 structured inferences/min
async def run_structured_inference(request):
    ...
```

---

## 🔵 Low Priority / Nice-to-Have

### 13. **Metrics Collection**
```python
@dataclass
class SynthesisMetrics:
    agent_response_times: List[float]
    total_retries: int
    validation_failures: int
    aggregation_time: float

# Return with results
return {
    'decision': ...,
    'metrics': metrics.dict()
}
```

### 14. **Circuit Breaker for External APIs**
```python
from pybreaker import CircuitBreaker

class OpenAIProvider:
    def __init__(self):
        self.breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

    def generate(self, prompt):
        return self.breaker.call(self._generate_impl, prompt)
```

### 15. **Prompt Length Validation**
```python
MAX_PROMPT_LENGTH = 100_000  # ~25K tokens

if len(prompt) > MAX_PROMPT_LENGTH:
    raise ValueError(f"Prompt exceeds max length {MAX_PROMPT_LENGTH}")
```

### 16. **Transaction Safety in Benchmark Creation**
```python
# Use temp key, only commit on success
temp_id = f"temp_{benchmark_id}"
try:
    self.index[temp_id] = data
    self._save_index()
    self.index[benchmark_id] = self.index.pop(temp_id)
    self._save_index()
except:
    self.index.pop(temp_id, None)
    raise
```

---

## Summary Priority Matrix

| Priority | Count | Effort | Impact |
|----------|-------|--------|--------|
| 🔴 Critical | 4 | 2-4h | High |
| 🟡 High | 5 | 4-8h | Medium-High |
| 🟢 Medium | 4 | 8-12h | Medium |
| 🔵 Low | 4 | 4-6h | Low-Medium |

**Total estimated effort: ~18-30 hours**

## Quick Wins (< 1 hour each)

1. Add empty checks before indexing (✅ #2, #8)
2. Add bounds validation to API endpoints (✅ #3)
3. Add basic logging statements (✅ #5)
4. Add options validation (✅ #7)

Implement these 4 first for immediate robustness improvement!
