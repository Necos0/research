from scipy.stats import pearsonr

def compute_pearson(x: list, y: list) -> tuple:
    corr_coeff, p_value = pearsonr(x, y)
    return corr_coeff, p_value

def get_correlation_strength(corr_coeff: float) -> str:
    """Return the correlation strength description based on the correlation coefficient."""
    if abs(corr_coeff) < 0.2:
        return "very weak"
    elif abs(corr_coeff) < 0.4:
        return "weak"
    elif abs(corr_coeff) < 0.6:
        return "moderate"
    elif abs(corr_coeff) < 0.8:
        return "strong"
    else:
        return "very strong"
    
def get_significance_level(p_value: float) -> str:
    """Return significance level based on the p-value."""
    if p_value <= 0.001:
        return "***"  # Very highly significant
    elif p_value <= 0.01:
        return "**"  # Highly significant
    elif p_value <= 0.05:
        return "*"  # Significant
    else:
        return "n.s."  # Not significant
    
def get_correlation_data(x: list, y: list) -> dict:
    corr_coeff, p_value = compute_pearson(x, y)
    correlation_strength = get_correlation_strength(corr_coeff)
    significance_level = get_significance_level(p_value)
    return {
        "correlation_coefficient": corr_coeff,
        "p_value": p_value,
        "correlation_strength": correlation_strength,
        "significance_level": significance_level
    }