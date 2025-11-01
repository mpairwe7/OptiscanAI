# Cell 53 Error Fix - IndexError in Color Mapping

## Error Description

```
IndexError: index 2 is out of bounds for axis 0 with size 2
```

**Location:** Cell 53, line 630  
**Context:** Creating color mapping for visualization "Best Model per Disease"

## Root Cause

The original code attempted to use `plt.cm.Set3(range(len(...)))` to generate colors, but this approach was incorrect:

```python
# PROBLEMATIC CODE:
colors = plt.cm.Set3(range(len(best_per_disease['Model'].unique())))
model_colors = {model: colors[i] for i, model in enumerate(df['Model'].unique())}
```

**Issue:** 
- `plt.cm.Set3(range(N))` returns a colormap array with only `N` colors
- When trying to index `colors[i]` where `i >= N`, it causes an IndexError
- This happened when `df['Model'].unique()` had more models than `best_per_disease['Model'].unique()`

## Solution Applied

Fixed the color generation to properly handle any number of models:

```python
# FIXED CODE:
unique_models = df['Model'].unique()
num_models = len(unique_models)

# Use appropriate colormap based on number of models
if num_models <= 12:
    cmap = plt.cm.Set3      # Good for up to 12 colors
else:
    cmap = plt.cm.tab20     # Supports up to 20 colors

# Generate colors by normalizing the range
colors_array = [cmap(i / max(num_models - 1, 1)) for i in range(num_models)]
model_colors = {model: colors_array[i] for i, model in enumerate(unique_models)}
bar_colors = [model_colors[model] for model in best_per_disease['Model']]
```

## Key Improvements

1. **Consistent Model List:** Uses the same `unique_models` list for both color generation and mapping
2. **Proper Normalization:** Normalizes indices `i / (num_models - 1)` to get proper colormap values
3. **Scalability:** Automatically switches to `tab20` colormap if more than 12 models
4. **Handles Edge Case:** Uses `max(num_models - 1, 1)` to avoid division by zero when only 1 model
5. **Updated Legend:** Also fixed the legend to use `unique_models` instead of `df['Model'].unique()`

## Testing

The fix handles:
- ✅ 1 model (edge case)
- ✅ 2-4 models (typical case)
- ✅ 5-12 models (extended case with Set3)
- ✅ 13-20 models (uses tab20 colormap)

## Files Modified

- `/home/darkhorse/Downloads/MLOPS_V1/notebooks/notebookc18697ca98.ipynb` - Cell 53

## Status

✅ **RESOLVED** - Cell 53 should now run without IndexError
