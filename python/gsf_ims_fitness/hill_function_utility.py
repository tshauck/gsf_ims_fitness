"""
This module contains functions to fit Hill functions to data.
"""


def hill_funct(x, low, high, mid, n):
    return low + (high - low) * (x**n) / (mid**n + x**n)


# Hill function of Hill function to describe fitness_difference(gene_expression([inducer]))
def double_hill_funct(x, g0, ginf, ec50, nx, f_min, f_max, g_50, ng):
    # g0, ginf, ec50, and nx are characteristics of individual sensor variants
    # g0 is the gene epxression level at zero inducer
    # ginf is the gene expresion level at full induction
    # ec50 is the inducer concentration of 1/2 max gene expression
    # nx is the exponent that describes the steepness of the sensor response curve
    # f_min, f_max, g_50, and ng are characteristics of the selection system
    # they are estimated from the fits above
    # f_min is the minimum fitness level, at zero gene expression
    # f_max is the maximum fitness level, at infinite gene expression (= 0)
    # g_50 is the gene expression of 1/2 max fitness
    # ng is the exponent that describes the steepness of the fitness vs. gene expression curve
    return hill_funct(hill_funct(x, g0, ginf, ec50, nx), f_min, f_max, g_50, ng)
