# hCaptcha Solver - Proposed Architecture

## Current Issues

1. **Monolithic `app/database.py`**: Handles all database operations for multiple collections (models, challenges, challenge_types, inferences, activity_logs, preprocess/postprocess profiles)
2. **Mixed Responsibilities**: Business logic mixed with HTTP handling in `api_gateway.py`
3. **No Service Layer**: Business logic scattered across API, database, and client modules
4. **No Domain Models**: Everything uses dictionaries, no type safety or validation
5. **Tight Coupling**: Direct database access from multiple layers
6. **No Repository Pattern**: Database queries scattered throughout codebase
7. **Validation Logic Scattered**: Challenge type validation in multiple places
8. **Large Streamlit File**: `main.py` has 1664 lines with mixed concerns

---

## Proposed Architecture: Layered Architecture with Domain-Driven Design

```
hcaptcha-solver/
├── app/                          # Core Application (Backend)
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   │
│   ├── domain/                   # Domain Models & Business Logic
│   │   ├── __init__.py
│   │   ├── models.py             # Domain models (Model, Challenge, ChallengeType, etc.)
│   │   ├── detection.py          # Detection result models
│   │   └── validators.py         # Domain validation rules
│   │
│   ├── repositories/             # Data Access Layer (Repository Pattern)
│   │   ├── __init__.py
│   │   ├── base.py               # Base repository interface
│   │   ├── model_repository.py   # Model CRUD operations
│   │   ├── challenge_repository.py
│   │   ├── challenge_type_repository.py
│   │   ├── inference_repository.py
│   │   ├── activity_log_repository.py
│   │   └── preprocess_repository.py
│   │
│   ├── services/                 # Business Logic Layer
│   │   ├── __init__.py
│   │   ├── model_service.py      # Model management, caching, loading
│   │   ├── inference_service.py  # Inference orchestration
│   │   ├── challenge_service.py  # Challenge processing, validation
│   │   ├── preprocessing_service.py
│   │   └── evaluation_service.py # Model evaluation logic
│   │
│   ├── inference/                # Inference Engine
│   │   ├── __init__.py
│   │   ├── solver.py             # YOLO inference (refactored from app/solver.py)
│   │   ├── cache.py              # Model caching logic
│   │   └── postprocessor.py     # Post-processing (NMS, filtering)
│   │
│   ├── preprocessing/            # Image Preprocessing
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Preprocessing pipeline (refactored from app/preprocess.py)
│   │   └── operations.py        # Individual preprocessing operations
│   │
│   ├── evaluation/               # Model Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation metrics (refactored from app/evaluator.py)
│   │   └── dataset_loader.py    # Dataset loading utilities
│   │
│   ├── api/                      # API Layer (HTTP)
│   │   ├── __init__.py
│   │   ├── routes.py             # Route definitions
│   │   ├── handlers/             # Request handlers
│   │   │   ├── __init__.py
│   │   │   ├── inference_handler.py
│   │   │   ├── model_handler.py
│   │   │   └── challenge_handler.py
│   │   ├── middleware.py        # Auth, logging, error handling
│   │   └── schemas.py            # Request/response schemas (validation)
│   │
│   ├── database/                 # Database Connection & Setup
│   │   ├── __init__.py
│   │   ├── connection.py         # MongoDB connection (refactored from app/database.py)
│   │   ├── gridfs_manager.py     # GridFS operations for model weights
│   │   └── indexes.py            # Database index definitions
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── image_utils.py        # Image conversion, compression (refactored from app/helper.py)
│       └── logging.py            # Structured logging
│
├── client/                       # Web Automation Clients
│   ├── __init__.py
│   ├── crawler/                  # Web Crawling
│   │   ├── __init__.py
│   │   ├── browser.py            # Browser setup and management
│   │   ├── extractor.py         # Challenge extraction (canvas/divs)
│   │   └── orchestrator.py       # Main crawl orchestration (refactored from crawler.py)
│   │
│   ├── clicker/                  # Click Automation
│   │   ├── __init__.py
│   │   ├── validator.py         # Detection validation (challenge-type-specific rules)
│   │   ├── canvas_clicker.py    # Canvas clicking logic
│   │   ├── tile_clicker.py      # Tile clicking logic
│   │   └── orchestrator.py      # Click orchestration (refactored from clicker.py)
│   │
│   └── api_client.py             # API client for sending challenges
│
├── streamlit_demo/               # Streamlit UI
│   ├── __init__.py
│   ├── main.py                   # Main entry point (simplified)
│   ├── pages/                    # Page modules
│   │   ├── __init__.py
│   │   ├── dataset_crawl.py
│   │   ├── eda.py
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   ├── model_management.py
│   │   ├── model_evaluation.py
│   │   └── demo.py
│   ├── components/               # Reusable UI components
│   │   ├── __init__.py
│   │   ├── model_selector.py
│   │   ├── image_display.py
│   │   └── metrics_charts.py
│   └── utils.py                  # Streamlit-specific utilities
│
├── tests/                        # Test Suite
│   ├── unit/
│   │   ├── domain/
│   │   ├── services/
│   │   ├── repositories/
│   │   └── inference/
│   ├── integration/
│   │   ├── api/
│   │   └── database/
│   └── e2e/
│       └── client/
│
└── scripts/                      # Utility Scripts
    ├── setup_db.py               # Database initialization
    └── seed_data.py              # Seed challenge types, models

```

---

## Layer Responsibilities

### 1. Domain Layer (`app/domain/`)
**Purpose**: Core business entities and rules

**Components**:
- **`models.py`**: Domain models (not database models)
  ```python
  @dataclass
  class Model:
      model_id: str
      model_name: str
      is_active: bool
      preprocess_id: Optional[str]
      postprocess_id: Optional[str]
      results: Dict[str, float]
  
  @dataclass
  class ChallengeType:
      challenge_type_id: str
      method: str  # "click" or "drag"
      keywords: List[str]
      model_id: Optional[str]
      validation_rules: Dict  # Challenge-specific validation rules
  
  @dataclass
  class Detection:
      class_name: str
      confidence: float
      bbox: List[float]  # [x1, y1, x2, y2]
  ```

- **`validators.py`**: Domain validation logic
  ```python
  class ChallengeTypeValidator:
      @staticmethod
      def validate_detections_for_challenge_type(
          detections: List[Detection],
          challenge_type: ChallengeType
      ) -> List[Detection]:
          """Apply challenge-type-specific validation rules."""
  ```

**Benefits**:
- Type safety
- Clear business rules
- Testable domain logic

---

### 2. Repository Layer (`app/repositories/`)
**Purpose**: Data access abstraction

**Pattern**: Repository Pattern
- Abstracts database operations
- Enables easy testing (mock repositories)
- Single responsibility per repository

**Example**:
```python
# app/repositories/base.py
class BaseRepository:
    def __init__(self, collection_name: str, db):
        self.collection = db[collection_name]
    
    def find_one(self, filter_dict: Dict) -> Optional[Dict]:
        return self.collection.find_one(filter_dict)
    
    def find_many(self, filter_dict: Dict, limit: int = 100) -> List[Dict]:
        return list(self.collection.find(filter_dict).limit(limit))
    
    def upsert(self, filter_dict: Dict, document: Dict) -> Dict:
        return self.collection.update_one(
            filter_dict, {"$set": document}, upsert=True
        )

# app/repositories/model_repository.py
class ModelRepository(BaseRepository):
    def __init__(self, db):
        super().__init__("model", db)
    
    def get_by_id(self, model_id: str) -> Optional[Dict]:
        return self.find_one({"model_id": model_id})
    
    def get_active(self) -> Optional[Dict]:
        return self.find_one({"is_active": True})
    
    def list_all(self, limit: int = 50) -> List[Dict]:
        return self.find_many({}, limit)
```

**Benefits**:
- Centralized data access
- Easy to swap database implementations
- Testable with mock repositories

---

### 3. Service Layer (`app/services/`)
**Purpose**: Business logic orchestration

**Responsibilities**:
- Coordinate between repositories
- Apply business rules
- Handle transactions
- Cache management

**Example**:
```python
# app/services/model_service.py
class ModelService:
    def __init__(
        self,
        model_repo: ModelRepository,
        gridfs_manager: GridFSManager,
        cache: ModelCache
    ):
        self.model_repo = model_repo
        self.gridfs_manager = gridfs_manager
        self.cache = cache
    
    def get_model_for_inference(self, challenge_type_id: str) -> Optional[Model]:
        """Get model for a challenge type, with caching."""
        challenge_type = self.challenge_type_repo.get_by_id(challenge_type_id)
        if not challenge_type:
            return None
        
        model_id = challenge_type.model_id
        if not model_id:
            return None
        
        # Check cache first
        model = self.cache.get(model_id)
        if model:
            return model
        
        # Load from database
        model_doc = self.model_repo.get_by_id(model_id)
        if not model_doc:
            return None
        
        # Load weights and cache
        weights_bytes = self.gridfs_manager.download_weights(model_id)
        model = self._load_model(model_doc, weights_bytes)
        self.cache.set(model_id, model)
        
        return model
```

**Benefits**:
- Single source of truth for business logic
- Reusable across API and client
- Easy to test

---

### 4. Inference Layer (`app/inference/`)
**Purpose**: Model inference engine

**Components**:
- **`solver.py`**: YOLO inference (refactored, cleaner interface)
- **`cache.py`**: Model caching logic (extracted from solver.py)
- **`postprocessor.py`**: NMS, confidence filtering

**Example**:
```python
# app/inference/solver.py
class InferenceEngine:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
    
    def infer(
        self,
        image: bytes,
        model_id: str,
        postprocess_config: Dict
    ) -> List[Detection]:
        """Run inference on image."""
        model = self.model_service.get_model_for_inference(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Convert image
        img_array = self._prepare_image(image)
        
        # Run inference
        results = model.predict(img_array, **postprocess_config)
        
        # Convert to domain models
        detections = self._parse_results(results)
        
        return detections
```

---

### 5. API Layer (`app/api/`)
**Purpose**: HTTP interface

**Structure**:
- **`routes.py`**: Route definitions
- **`handlers/`**: Request handlers (thin, delegate to services)
- **`schemas.py`**: Request/response validation (using Pydantic or similar)

**Example**:
```python
# app/api/handlers/inference_handler.py
class InferenceHandler:
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
    
    def handle_solve_hcaptcha(self, request: Request) -> Response:
        """Handle POST /solve_hcaptcha."""
        # Validate request
        image = request.files['image']
        question = request.form.get('question')
        
        # Delegate to service
        result = self.inference_service.process_challenge(
            image_bytes=image.read(),
            question=question
        )
        
        # Format response
        return jsonify(result)
```

**Benefits**:
- Thin controllers
- Easy to add new endpoints
- Clear separation of HTTP and business logic

---

### 6. Client Layer (`client/`)
**Purpose**: Web automation

**Structure**:
- **`crawler/`**: Web crawling logic
- **`clicker/`**: Click automation logic
- **`api_client.py`**: API communication

**Improvements**:
- Extract browser management to `browser.py`
- Separate extraction logic from orchestration
- Move validation to `clicker/validator.py` (already done)

---

### 7. Streamlit Demo (`streamlit_demo/`)
**Purpose**: UI for testing and management

**Structure**:
- **`pages/`**: Separate page modules
- **`components/`**: Reusable UI components
- **`main.py`**: Simplified routing

**Benefits**:
- Easier to maintain
- Reusable components
- Clear page boundaries

---

## Migration Strategy

### Phase 1: Extract Repositories (Low Risk)
1. Create `app/repositories/` structure
2. Move database operations from `app/database.py` to repositories
3. Update `app/database.py` to use repositories internally
4. **No breaking changes** - existing code still works

### Phase 2: Create Service Layer (Medium Risk)
1. Create `app/services/` structure
2. Move business logic from `api_gateway.py` to services
3. Update API handlers to use services
4. **Gradual migration** - migrate one endpoint at a time

### Phase 3: Refactor Inference & Preprocessing (Medium Risk)
1. Move inference logic to `app/inference/`
2. Move preprocessing to `app/preprocessing/`
3. Update services to use new modules

### Phase 4: Create Domain Models (Low Risk, High Value)
1. Create domain models in `app/domain/`
2. Gradually replace dict usage with domain models
3. Add validation logic

### Phase 5: Refactor Streamlit (Low Risk)
1. Split `main.py` into page modules
2. Extract reusable components
3. **No functional changes**

### Phase 6: Refactor Client (Low Risk)
1. Split `crawler.py` into modules
2. Split `clicker.py` into modules
3. Extract API client

---

## Key Design Principles

### 1. Dependency Injection
```python
# Instead of:
def solve_captcha(image, question, config):
    model = load_model_from_db(config['model_id'])  # Hard dependency

# Use:
class InferenceService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service  # Injected dependency
```

### 2. Single Responsibility
- Each module/class has one clear purpose
- Repositories: Data access only
- Services: Business logic only
- Handlers: HTTP handling only

### 3. Dependency Direction
```
API → Services → Repositories → Database
     ↓
  Domain Models
```

### 4. Testability
- Services can be tested with mock repositories
- Repositories can be tested with in-memory databases
- Domain logic is pure functions (easy to test)

---

## Benefits of Proposed Architecture

1. **Maintainability**: Clear separation of concerns
2. **Testability**: Each layer can be tested independently
3. **Scalability**: Easy to add new features
4. **Type Safety**: Domain models provide type hints
5. **Reusability**: Services can be used by API, client, and Streamlit
6. **Flexibility**: Easy to swap implementations (e.g., different database)

---

## File Size Reduction

**Current**:
- `app/database.py`: 633 lines (multiple responsibilities)
- `app/api_gateway.py`: 618 lines (mixed concerns)
- `streamlit_demo/main.py`: 1664 lines (all pages in one file)

**After Refactoring**:
- Each repository: ~100-150 lines
- Each service: ~150-200 lines
- Each API handler: ~50-100 lines
- Each Streamlit page: ~200-300 lines

---

## Next Steps

1. **Review this proposal** with the team
2. **Start with Phase 1** (Repositories) - lowest risk
3. **Create domain models** for new features
4. **Gradually migrate** existing code
5. **Add tests** as you refactor

---

## Questions to Consider

1. Should we use a validation library (Pydantic) for domain models?
2. Should we add dependency injection framework (e.g., dependency-injector)?
3. Should we add structured logging framework?
4. Should we add API documentation (OpenAPI/Swagger)?
5. Should we add database migrations framework?

