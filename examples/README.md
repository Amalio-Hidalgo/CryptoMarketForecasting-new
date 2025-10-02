# Examples & Tutorials

Cross-project examples demonstrating different use cases and comparisons between V1 and V2.

## 🎯 Quick Navigation

| Example Type | Skill Level | Time | Purpose |
|--------------|-------------|------|---------|
| **[Basic Usage](basic-usage/)** | Beginner | 15 min | Get started quickly |
| **[Advanced Scenarios](advanced-scenarios/)** | Intermediate | 45 min | Real-world applications |
| **[Migration Examples](migration-examples/)** | Advanced | 30 min | V1 → V2 transition |
| **[Comparison Studies](comparison-studies/)** | All levels | 20 min | V1 vs V2 analysis |

## 📚 Example Categories

### 🚀 Basic Usage
Perfect for getting started or teaching others:

- **`quick-start-v2.ipynb`**: 5-minute volatility prediction
- **`data-collection-demo.ipynb`**: Multi-API data gathering
- **`simple-prediction.ipynb`**: End-to-end basic workflow
- **`configuration-examples.ipynb`**: Different setup scenarios

### 🏭 Advanced Scenarios  
Real-world applications and complex use cases:

- **`portfolio-risk-management.ipynb`**: Multi-crypto volatility analysis
- **`market-regime-detection.ipynb`**: Using volatility for market classification
- **`real-time-monitoring.ipynb`**: Live volatility tracking
- **`custom-features.ipynb`**: Adding domain-specific indicators

### 🔄 Migration Examples
For users transitioning from V1 to V2:

- **`v1-to-v2-data-migration.ipynb`**: Convert existing datasets
- **`feature-comparison.ipynb`**: V1 vs V2 feature engineering
- **`model-performance-comparison.ipynb`**: Side-by-side results
- **`workflow-modernization.ipynb`**: Updating processes

### 📊 Comparison Studies
Analytical comparisons between approaches:

- **`volatility-vs-price-targets.ipynb`**: Target comparison analysis
- **`single-vs-multi-source.ipynb`**: Data source impact study
- **`manual-vs-automated.ipynb`**: Process efficiency analysis
- **`cost-benefit-analysis.ipynb`**: V1 vs V2 operational costs

## 🎓 Learning Path

### Path 1: Complete Beginner
1. Start with `basic-usage/quick-start-v2.ipynb`
2. Try `basic-usage/data-collection-demo.ipynb`
3. Explore `comparison-studies/volatility-vs-price-targets.ipynb`
4. Graduate to `advanced-scenarios/portfolio-risk-management.ipynb`

### Path 2: V1 User Migrating
1. Review `comparison-studies/v1-vs-v2-overview.ipynb`
2. Work through `migration-examples/v1-to-v2-data-migration.ipynb`
3. Compare results with `migration-examples/model-performance-comparison.ipynb`
4. Implement `migration-examples/workflow-modernization.ipynb`

### Path 3: Advanced Practitioner
1. Dive into `advanced-scenarios/custom-features.ipynb`
2. Implement `advanced-scenarios/real-time-monitoring.ipynb`
3. Study `comparison-studies/cost-benefit-analysis.ipynb`
4. Contribute your own examples!

## 🔧 Running Examples

### Prerequisites
```bash
# For V2 examples
cd ../v2-volatility-forecasting
pip install -e .

# For V1 examples (if needed)
cd ../v1-price-forecasting  
pip install -r requirements.txt

# For comparison examples
# Both V1 and V2 dependencies needed
```

### Environment Setup
```bash
# Copy environment template
cp ../v2-volatility-forecasting/.env.example .env

# Edit with your API keys
# COINGECKO_API_KEY=your_key_here
# DUNE_API_KEY_2=your_key_here
# FRED_API_KEY=your_key_here
```

### Running Notebooks
```bash
# Start Jupyter
jupyter lab

# Or run specific notebook
jupyter nbconvert --execute example.ipynb
```

## 📝 Example Template

When creating new examples, use this structure:

```python
"""
Example: [Title]
Difficulty: [Beginner/Intermediate/Advanced]
Time: [XX minutes]
Purpose: [What this demonstrates]

Requirements:
- API keys: [which ones needed]
- Data: [any special data requirements]
- Time: [if real-time data needed]
"""

# 1. Setup and imports
import sys
sys.path.append('../v2-volatility-forecasting/src')

# 2. Configuration
config = {
    'target_coin': 'ethereum',
    'lookback_days': 30,
    'frequency': '1D'
}

# 3. Main example code
[...demonstration...]

# 4. Results and interpretation
print("Results:")
print("- Key finding 1")
print("- Key finding 2")

# 5. Next steps
print("\nNext steps:")
print("- Try with different parameters")
print("- Explore related examples")
```

## 🤝 Contributing Examples

We welcome new examples! Please:

1. **Follow the template** above
2. **Test thoroughly** before submitting
3. **Document clearly** what the example shows
4. **Include expected outputs** or screenshots
5. **Add to the navigation table** above

### Example Ideas Needed
- [ ] Sector-specific crypto analysis
- [ ] Integration with trading platforms
- [ ] Custom visualization dashboards
- [ ] Model explainability examples
- [ ] Performance benchmarking
- [ ] Cloud deployment examples

## 🐛 Reporting Issues

If you find problems with examples:

1. **Check Prerequisites**: Ensure all dependencies installed
2. **Verify API Keys**: Make sure your .env file is configured
3. **Check Data Availability**: Some examples need recent data
4. **Open Issue**: Include notebook name and error message

---

**💡 Have an idea for a useful example?** Open an issue or submit a pull request!