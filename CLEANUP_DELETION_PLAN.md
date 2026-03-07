# 🧹 SmartMonitor API - Model Management Cleanup Plan

**Date:** March 7, 2026  
**Purpose:** Remove unused model management infrastructure (caching, repositories, cloud storage)

---

## 📊 Current State Analysis

### ✅ ACTIVE Endpoints (Must Keep Working)
1. **PredicaoMensal** - Uses `PredicaoService` (trains models on-the-fly)
2. **PredicaoDiaria** - Uses `PredicaoService` (trains models on-the-fly)
3. **Analise_estatistica_mensal** - Uses `AnaliseEstatisticaService` (pure statistics)
4. **Analise_estatistica_diaria** - Uses `AnaliseEstatisticaService` (pure statistics)
5. **DadosBandas** - Uses `dadosBandas_service` (pure statistics)
6. **ClassificacaoPH** - Uses `PHClassificationService` (loads pre-trained models)

### 🔍 Key Findings

#### Services That DON'T Need Model Loading:
- ✅ `PredicaoService` - Creates and trains `LinearRegression_Acumulado` in-memory with request data
- ✅ `AnaliseEstatisticaService` - Pure statistical analysis (Bollinger Bands), no ML models
- ✅ `dadosBandas_service` - Pure statistical calculations, no ML models

#### Service That DOES Need Model Loading:
- ⚠️ `PHClassificationService` - Loads pre-trained scikit-learn models from disk
  - Currently uses complex `model_repository` → `model_cache` infrastructure
  - Only needs simple `joblib.load()` from local files

---

## 🗑️ Phase 1: Files to DELETE

### 1.1 Model Management Infrastructure
```
✗ ml_pipeline/model_cache.py (207 lines)
  - Defines ModelCacheManager with TTL-based caching
  - Only used by model_repository
  - NOT NEEDED: Simple direct loading is faster and simpler

✗ ml_pipeline/model_repository/__init__.py (298 lines)
  - Complex ModelRepository with cache → disk → Drive flow
  - DriveModelStorage class (not implemented)
  - Only used by PHClassificationService
  - REPLACEMENT: Direct joblib.load() in service

✗ ml_pipeline/retreino.py (75 lines)
  - ModelRetrainingService class (all methods raise NotImplementedError)
  - NOT imported or used anywhere
  - Future feature that was never needed
```

### 1.2 Documentation (Outdated)
```
✗ docsmd/FLUXO_API_PH_CLASSIFICATION.md
  - Describes complex cache/repository/drive architecture
  - No longer applicable after simplification

✗ docsmd/GERENCIAMENTO_MODELOS_DRIVE.md
  - Describes Google Drive integration
  - Never implemented, not needed

✗ docsmd/RESUMO_PH_CLASSIFICATION.md
  - Summary of complex architecture
  - Will be outdated after cleanup

✗ docsmd/TESTE_PH_CLASSIFICATION.md
  - Tests for complex infrastructure
  - Will be outdated after cleanup
```

### 1.3 Credentials (Unused)
```
✗ credentials/smartmonitorapi-478917-b1b4e690a32c.json
  - Google Drive API credentials
  - Never used (GOOGLE_DRIVE_ENABLED = False)
  - Security risk to keep unused credentials

✗ credentials/ (entire directory)
  - Can be removed after removing the JSON file
```

---

## ♻️ Phase 2: Files to SIMPLIFY

### 2.1 PHClassificationService
**File:** `ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py`

**Current Dependencies:**
```python
from ml_pipeline.model_repository import model_repository
```

**Changes:**
1. Remove `model_repository` import
2. Add direct `joblib` import
3. Simplify `classify()` method to:
   - Build path to model file directly
   - Use `joblib.load(model_path)` directly
   - Remove cache retrieval logic
4. Remove or simplify `get_model_info()` method
5. Keep model path structure: `MODELS_DIR/ph_classification/client_{client_id}/model_v*.joblib`

**Before (complex):** 73 lines with repository abstraction  
**After (simple):** ~40 lines with direct loading

### 2.2 Django Settings
**File:** `projectSM/settings.py`

**Remove:**
```python
# Cache de modelos ML
MODEL_CACHE_TTL = config('MODEL_CACHE_TTL', default=60, cast=int)  # minutos

# Google Drive (opcional - para futuro)
GOOGLE_DRIVE_ENABLED = config('GOOGLE_DRIVE_ENABLED', default=False, cast=bool)
GOOGLE_DRIVE_CREDENTIALS = config('GOOGLE_DRIVE_CREDENTIALS', default='')
```

**Keep:**
```python
# Configuração de modelos ML
MODELS_DIR = BASE_DIR / 'ml_pipeline' / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)
```

### 2.3 API View Documentation
**File:** `appSM/views.py`

**In `ClassificacaoPH` docstring, change:**
```python
# OLD:
"2. Carrega modelo do cliente (cache → disco → Google Drive)\n"

# NEW:
"2. Carrega modelo do cliente do disco local\n"
```

### 2.4 README
**File:** `README.md`

**Update sections to remove mentions of:**
- Model caching
- Google Drive integration
- Multi-tenant repository architecture

**Keep mentions of:**
- Local model storage
- Model versioning (simple file-based)

---

## 📦 Phase 3: Dependencies to REMOVE

### 3.1 Python Packages (requirements.txt)
```
❌ NO REMOVALS NEEDED

Reason: All packages are still required:
- joblib: Used by PHClassificationService for direct model loading
- scikit-learn: Used by LinearRegression_Acumulado and loaded pH models
- All other dependencies are Django/DRF related
```

---

## 🏗️ Phase 4: Architecture Changes

### Before (Complex - 3-tier loading)
```
Request
  ↓
PHClassificationService
  ↓
ModelRepository.load_model()
  ↓
┌─ Check ModelCache (memory)
│  ├─ Hit → return cached model
│  └─ Miss ↓
├─ Check Local Disk
│  ├─ Found → load with joblib → cache → return
│  └─ Not Found ↓
└─ Check Google Drive (not implemented)
   ├─ Found → download → save to disk → cache → return
   └─ Not Found → raise FileNotFoundError
```

### After (Simple - direct loading)
```
Request
  ↓
PHClassificationService.classify()
  ↓
├─ Build path: MODELS_DIR/ph_classification/client_{id}/model_v*.joblib
├─ Use joblib.load(model_path) directly
├─ Make prediction
└─ Return result
```

**Benefits:**
- ✅ 80% less code
- ✅ No cache invalidation complexity
- ✅ Easier to debug (no hidden layers)
- ✅ Models still on disk (versioning preserved)
- ✅ Fast enough (joblib is already optimized)
- ✅ No unused infrastructure
- ✅ Security improved (no unused credentials)

---

## ⚠️ Phase 5: Potential Risks & Mitigation

### Risk 1: Performance Degradation
**Concern:** Loading model from disk on every request vs. cached in memory

**Analysis:**
- Current cache: Loads once, stores in RAM (TTL = 60 min)
- New approach: Loads from disk each time
- joblib.load() on modern SSD: ~50-200ms (acceptable for API)
- Model size: Typically 1-10 MB (small)

**Mitigation:**
- ✅ Keep model files small (already the case)
- ✅ If performance issue arises later, add simple Python @lru_cache decorator
- ✅ Monitor response times after deployment

**Decision:** Risk is LOW - acceptable tradeoff for simplicity

### Risk 2: Breaking Existing pH Classification Functionality
**Concern:** PHClassificationService might stop working

**Mitigation:**
- ✅ Keep exact same model file structure
- ✅ Test endpoint before/after with same request
- ✅ Model path logic stays identical (just simplified)
- ✅ Keep metadata JSON files for documentation

**Decision:** Risk is VERY LOW - straightforward refactor

### Risk 3: Future Scalability
**Concern:** What if we need caching/drive later?

**Mitigation:**
- ✅ YAGNI principle: Don't build what you don't need
- ✅ Can add back later if actually needed
- ✅ Current code is over-engineered for actual usage
- ✅ Simpler base = easier to extend when needed

**Decision:** Risk is LOW - premature optimization removed

### Risk 4: Multi-client Model Management
**Concern:** Managing models for multiple clients becomes harder

**Analysis:**
- Current structure already file-based: `client_sisar/model_v1.0.0.joblib`
- Only 1 client exists (sisar)
- Adding new clients: Just create new folder with model file

**Mitigation:**
- ✅ Keep folder structure: `ph_classification/client_{id}/`
- ✅ Keep version naming: `model_v1.0.0.joblib`
- ✅ Keep metadata JSON for documentation

**Decision:** Risk is MINIMAL - simpler is better for file management

---

## 📋 Phase 6: Implementation Order

### Step 1: Create Backup
```bash
git checkout -b cleanup/remove-model-management
git add -A
git commit -m "Checkpoint before cleanup"
```

### Step 2: Remove Unused Files (Safe - not imported)
1. Delete `ml_pipeline/retreino.py` ✓
2. Delete `credentials/smartmonitorapi-478917-b1b4e690a32c.json` ✓
3. Delete `credentials/` directory ✓
4. Delete all files in `docsmd/` ✓

### Step 3: Simplify PHClassificationService
1. Modify `ml_pipeline/senseflowQ/ph_classification/ph_classification_service.py`
2. Remove `model_repository` import
3. Add `joblib` import and settings import
4. Rewrite `classify()` method with direct loading
5. Optionally simplify `get_model_info()` method

### Step 4: Remove Infrastructure (Now Safe)
1. Delete `ml_pipeline/model_cache.py` ✓
2. Delete `ml_pipeline/model_repository/__init__.py` ✓
3. Delete `ml_pipeline/model_repository/` directory ✓

### Step 5: Clean Up Settings
1. Modify `projectSM/settings.py`
2. Remove Google Drive and cache TTL config

### Step 6: Update Documentation
1. Update `appSM/views.py` docstring
2. Update `README.md` (remove old architecture mentions)

### Step 7: Test
1. Start Django server
2. Test `/classify/ph` endpoint with sample data
3. Verify all 6 endpoints still work
4. Check logs for any errors

### Step 8: Finalize
```bash
git add -A
git commit -m "Remove unused model management infrastructure"
```

---

## 📊 Summary Statistics

### Code Reduction
| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| model_cache.py | 207 lines | DELETED | -207 lines |
| model_repository/__init__.py | 298 lines | DELETED | -298 lines |
| retreino.py | 75 lines | DELETED | -75 lines |
| ph_classification_service.py | 180 lines | ~80 lines | -100 lines |
| settings.py | ~150 lines | ~145 lines | -5 lines |
| **TOTAL** | **910 lines** | **225 lines** | **-685 lines (75%)** |

### Files Changed
- **Deleted:** 9 files (3 Python modules + 4 docs + 1 credential + 1 dir)
- **Modified:** 4 files (service, settings, views, README)
- **Kept:** All model files, all active services

### Architecture Complexity
- **Before:** 3-tier loading system (cache → disk → cloud)
- **After:** Direct loading (simple joblib.load)
- **Complexity Reduction:** ~80%

---

## ✅ Success Criteria

### Must Have:
1. ✅ All 6 API endpoints work correctly
2. ✅ pH classification still functions (loads models)
3. ✅ No imports from deleted modules
4. ✅ All tests pass (if tests exist)
5. ✅ Django starts without errors

### Nice to Have:
1. ✅ Response times similar or better
2. ✅ Logs are cleaner (less infrastructure noise)
3. ✅ Code is easier to understand
4. ✅ Future changes easier to implement

---

## 🎯 Next Steps

**Ready for approval?** Review this plan and confirm before proceeding.

**After approval:**
1. Execute Step 1 (backup)
2. Execute Steps 2-6 (deletions and modifications)
3. Execute Step 7 (testing)
4. Report results
5. Execute Step 8 (commit) if successful

**Questions to address:**
- Should we keep the metadata JSON files? (Recommendation: YES - documentation value)
- Should we add a simple in-memory cache later if needed? (Recommendation: Only if performance requires)
- Should we create a new minimal documentation file? (Recommendation: Update README only)

---

**END OF DELETION PLAN**
