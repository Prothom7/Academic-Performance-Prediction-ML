# Reference-paper comparison

The local [reference PDF](../references/academic_performance_paper.pdf) is Delahoz-Dominguez, Zuluaga and Fontalvo-Herrera (2020), *Dataset of academic performance evolution for engineering students*, Data in Brief 30, 105537, [DOI](https://doi.org/10.1016/j.dib.2020.105537). Its population size and variable count below are extracted and checked by `src/research/documentation.py`, with the extracted text retained in [the analysis log](../results/logs/reference_paper_extracted.txt).

| dimension           | reference_dataset                                       | current_project                                                                     |
|:--------------------|:--------------------------------------------------------|:------------------------------------------------------------------------------------|
| Population size     | 12411                                                   | 3516                                                                                |
| Variables           | 44                                                      | 32                                                                                  |
| Design              | Linked Colombian ICFES assessment databases             | Current assembled academic-performance survey population                            |
| Educational stages  | Secondary-school Saber 11 and end-of-training Saber Pro | HSC preparation plus Year-1 and Year-2 CGPA bands                                   |
| Academic evolution  | Competencies assessed across educational stages         | Ordinal persistence, transition matrix and mobility categories                      |
| Context             | Academic, social and economic variables                 | Academic, socioeconomic, institutional, preferences, living and financial variables |
| Research engagement | Not an outcome described in the article                 | Binary research participation; extracurricular and contextual associations          |
| Comparison scope    | Data-description article                                | Predictive and explanatory association analyses; no accuracy-superiority claim      |

These datasets concern different populations, educational systems, stages and measurements. Neither accuracy nor distributions constitute a direct population comparison. The current project adds explicit university-band mobility and research-engagement analyses; it does not establish superiority over the reference article. See [reference dataset comparison](../results/tables/reference_dataset_comparison.csv) for a machine-readable comparison.
