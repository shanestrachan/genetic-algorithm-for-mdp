# Matrix Domination: Convergence of a Genetic Algorithm Metaheuristic with the Wisdom of Crowds to Solve the NP-Complete Problem

**Project and Paper Author:** Shane Storm Strachan

*Full Paper Forthcoming*

## Abstract

This research explores the application of a genetic algorithm metaheuristic, enhanced through collective intelligence through the "wisdom of crowds," to tackle the NP-Complete Matrix Domination Problem (henceforth, TMDP). TMDP is identified as a specialized constraint problem where the objective is to strategically place a subset of nodes, termed dominators, within a matrix. The goal is for these dominators to exert control over the remaining nodes, a concept akin to achieving comprehensive influence within a graph. By leveraging the inherent exploratory nature of genetic algorithms and combining it with the wisdom of crowds, allows the algorithm to more effectively navigate the complex landscape possibilities of TMDP, especially in larger matrix sizes. My methodology relies on the employment of a fitness evaluation function to measure the efficacy of current solutions and a constraining function to address the stochastic characteristics typical of genetic algorithms and keep the solutions within the NP-Complete problem's definition.

A novel aspect of this work is the proposed approach to TMDP using a genetic algorithm, moreover one that harnesses collective decision-making in its selection process. The exploration process is pivotal in uncovering the most effective solutions for TMDP. Findings demonstrate the effectiveness of this hybrid approach in optimizing the balance between the quantity of dominators and their strategic positioning within the matrix. This optimization ensures both efficient and comprehensive control over the matrix, a key goal in graph and matrix-based problems.

### Keywords

NP-complete, graph theory, matrix domination, genetic algorithm, wisdom of crowds

## Introduction

Matrix Domination, recognized as an NP-Complete problem, was first identified over four decades ago. This problem intricately intertwines with fundamental aspects of graph theory, particularly emphasizing graph edge traversal and domination concepts. In the realm of graph theory, the focus often lies on the analysis of dominating sets, which are primarily concerned with vertices and their immediate neighbors. Dominating sets are instrumental in understanding the control or influence exerted by a set of vertices over the entire graph.

Matrix Domination extrapolates these principles into a matrix context. Here, the elements of a matrix are not just standalone entities; instead, they are analogues to vertices in a graph. The relationships between these matrix elements mirror the connections between vertices in a graph. The essence of TMDP lies in strategically placing a subset of these matrix cells, termed dominators, to effectively dominate (in other words, influence) the rest of the matrix. This strategic placement is akin to determining a set of vertices in a graph that can exert their influence over the entire graph structure. The challenge in Matrix Domination, much like in identifying dominating sets in graph theory, is to achieve this domination with the smallest number of dominators and as quickly as possible, reflecting efficiency and optimization in both contexts. Thus, Matrix Domination is not only a problem of computational significance due to its NP-Completeness, but is also important to matrix theory and graph theory. It extends the conceptual framework of graph domination into a matrix setting, thereby offering a unique perspective to explore and solve its problem sets.

### The Matrix Domination Problem (TMDP)

The primary objective of Matrix Domination regards the strategic deployment of the minimal number of dominators across the matrix. This minimalistic approach is crucial, as it directly correlates with the efficiency and efficacy of the domination process. In this context, a matrix cell is seen as dominated under two scenarios: firstly, if it houses a dominator itself, and secondly, if it lies orthogonally adjacent to a cell containing a dominator. This criterion for domination underscores a multifaceted relationship between matrix cells, which remains intricate even in smaller matrices, emphasizing the subtleties and complexities inherent in the problem.

The practical implications extend far and wide, relevant to various domain applications. It finds significant applications in network design, where the concept of domination can parallel strategies in network robustness and redundancy. In logistics, the principles aid in optimizing resource distribution and route planning. Similarly, in surveillance systems, the ideas translate into effective monitoring strategies, and in resource allocation, they assist in maximizing coverage with minimal resources. Each of these applications demonstrates the versatility and utility of Matrix Domination principles in addressing complex organizational and operational scenarios.

While exploring Matrix Domination, the focus predominantly rests on the binary dynamic of dominators and the dominated, sidestepping the additional layers of complexity that 'tiers' of domination might introduce. Such an approach aligns with the theoretical essence of the problem, allowing for a more concentrated examination of its core elements.

However, the NP-completeness of TMDP is a testament to its computational intensity, particularly when dealing with large or intricately structured matrices. This complexity necessitates the use of advanced problem-solving strategies, like metaheuristics or approximation algorithms. These methods are not straightforward solutions but rather sophisticated approaches designed to navigate the labyrinth of possibilities in search of an optimal or near-optimal solution. Their employment is essential in managing the computational demands and inherent uncertainties of complete domination.

### Formal Problem Statement

The problem can be explicitly stated as follows:

Let there be an n x n matrix M with entries from { 0, 1 }, and a positive integer K. Is there a set of K, or fewer, non-zero entries in M that dominate all others? In other words, can s subset C ⊆ { 1, 2, ... , n } x { 1, 2, ... , n } with |C| ≤ K such that Mij = 1 for all (i, j) ∈ C and such that, whenever Mij = 1, there exists an (i’, j’) ∈ C for which either i = i’ or j = j’?

The core challenge presented in Matrix Domination revolves around a specific n x n binary matrix M, with its entries limited to the binary set {0, 1}. Accompanying this matrix is a positive integer K, which serves as a crucial parameter in the problem. The principal query is whether there exists a subset, denoted as C, within this matrix, comprising a specific arrangement of indices (each representing a coordinate point defined by row and column numbers). This subset C is constrained in size by the integer K, implying that it should contain K or fewer elements.

Each element within C, represented as a pair of indices (i, j), corresponds to a non-zero (or '1') entry in matrix M. The concept of domination is defined by a relational rule: for every '1' present in the matrix M, there must be a corresponding element in subset C that shares either the same row (i = i’) or the same column (j = j’). This relational rule establishes a form of control or influence exerted by elements in C over the matrix, reflecting the notion of domination. Furthermore, the problem maintains its computational complexity, classified as NP-complete, regardless of the specific form of matrix M. This complexity persists even in specialized matrix configurations, such as upper triangular matrices. The persistence of NP-completeness in these specialized cases underscores inherent complexity and computational challenges and it highlights the intricate nature of the problem, where the simplicity of the matrix's structure does not necessarily translate to a simplification of the problem's resolution.

In summary, TMDP is an exploration of strategic placement and influence within a binary matrix, governed by specific rules and constraints. It encapsulates a challenging intersection of combinatorial optimization and graph theory, where the objective is to find an optimal or near-optimal subset that adheres to the domination rule, within the bounds of computational feasibility.

### Suggested Further Reading

M Garey and D Johnson, 1979. *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman Publishing.

M. Yannakakis and F. Gavril, 1980. “Edge Dominating Sets in Graphs” *SIAM Journal on Applied Mathematics*, 38(3), pp. 364-372.

J Horton and K Kilakos, 1993, “Minimum Edge Dominating Sets” *SIAM Journal on Discrete Mathematics* 6(3),  pp. 375-387.

T Haynes, S Hedetniemi, and P Slater, 1998. *Fundamentals of Domination in Graphs*. Marcel Dekker Publishing. 

R Yampolskiy, L Ashby, and L Hassan, 2012. “Wisdom of Artificial Crowds–A Metaheuristic Algorithm for Optimization,” *Journal of Intelligent Learning Systems and Applications* 4, pp. 98-107.

M Henning and A Yeo, 2013. *Total Domination in Graphs*, Springer Publishing. 
