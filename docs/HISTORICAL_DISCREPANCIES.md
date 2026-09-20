# Historical checkpoint comparison

Generated comparison: [historical checkpoint comparison](../results/tables/historical_checkpoint_comparison.csv). 32 checkpoints differ beyond the declared approximate tolerance. Each row retains the expected checkpoint, computed result, difference, denominator/configuration context and authoritative output path. These are research changes or unresolved historical-specification differences, not targets for numerical tuning.

The exact historical WEKA API refit matches saved-model predictions and probabilities. A separate CLI route initially produced different output. The preserved model contains historically corrupted bilingual symbolic labels. WEKA's [data-dependent RNG implementation](https://github.com/Waikato/weka-3.8/blob/master/weka/src/main/java/weka/core/Instances.java) includes a row-string hash, so symbolic text can change randomized training even when numeric nominal indices and the explicit seed match. The replay helper verifies zero input-cell differences and records different derived RNG values for the original versus UTF-8 source headers. It aligns the current source to the unique English prefixes of the preserved nominal header, restores the original symbolic encoding, and independently refits; no predictors, outcomes, rows or configuration are tuned to checkpoints. The CLI diagnostic remains in [its JSON](../results/metrics/research_historical_weka_cli_diagnostic.json) and [executed log](../results/logs/research_historical_weka.txt).

Year-2 uses the newly documented directly comparable feature-set protocol because no original Year-2 executable/configuration is present. Adjusted and interaction specifications include all measured background variables, ordinal trends and documented contextual controls; historical specifications and test types are unavailable. Those configuration differences remain unresolved rather than being retrospectively reconstructed to target p-values. Unadjusted research tests use the binary outcome; any historical multi-category target may therefore differ. Missingness denominators and Faculty ECE pooling are explicit in the model diagnostics. Coefficient CIs and interaction evidence are taken from the computed tables.

| checkpoint                                                                     |   historical_expected |     computed |   difference | within_tolerance   |
|:-------------------------------------------------------------------------------|----------------------:|-------------:|-------------:|:-------------------|
| year1/accuracy                                                                 |               0.5985  |  0.598485    | -1.51515e-05 | True               |
| year1/weighted_f1                                                              |               0.5984  |  0.59837     | -2.99188e-05 | True               |
| year1/macro_f1                                                                 |               0.6033  |  0.603309    |  8.77769e-06 | True               |
| year2_without_year1/accuracy                                                   |               0.5587  |  0.649621    |  0.0909212   | False              |
| year2_without_year1/weighted_f1                                                |               0.5569  |  0.646766    |  0.0898663   | False              |
| year2_without_year1/macro_f1                                                   |               0.567   |  0.657557    |  0.090557    | False              |
| year2_with_year1/accuracy                                                      |               0.8712  |  0.912879    |  0.0416788   | False              |
| year2_with_year1/weighted_f1                                                   |               0.8713  |  0.913085    |  0.0417845   | False              |
| year2_with_year1/macro_f1                                                      |               0.8736  |  0.907935    |  0.0343349   | False              |
| research_historical_weka/accuracy                                              |               0.8103  |  0.810345    |  4.48276e-05 | True               |
| research_historical_weka/participated_f1                                       |               0.699   |  0.699088    |  8.81459e-05 | True               |
| trajectory/Stable                                                              |              75       | 75           |  0           | True               |
| trajectory/Improved                                                            |              10.84    | 10.8362      | -0.00382253  | True               |
| trajectory/Declined                                                            |              14.16    | 14.1638      |  0.00382253  | True               |
| persistence/spearman                                                           |               0.847   |  0.84681     | -0.000190073 | True               |
| persistence/cramers_v                                                          |               0.695   |  0.695379    |  0.000378772 | True               |
| CramersV/HSCMath/Year1Band                                                     |               0.17    |  0.169702    | -0.000297516 | True               |
| CramersV/HSCMath/Year2Band                                                     |               0.156   |  0.156397    |  0.000397112 | True               |
| CramersV/HSCMath/Trajectory                                                    |               0.033   |  0.0330969   |  9.69224e-05 | True               |
| CramersV/HSCMath/research                                                      |               0.109   |  0.108604    | -0.00039579  | True               |
| CramersV/HSCPhysics/Year1Band                                                  |               0.155   |  0.154832    | -0.000167583 | True               |
| CramersV/HSCPhysics/Year2Band                                                  |               0.134   |  0.13384     | -0.000159884 | True               |
| CramersV/HSCPhysics/Trajectory                                                 |               0.06    |  0.0595462   | -0.000453812 | True               |
| CramersV/HSCPhysics/research                                                   |               0.104   |  0.103889    | -0.000111315 | True               |
| CramersV/HSCChemistry/Year1Band                                                |               0.149   |  0.149006    |  6.13419e-06 | True               |
| CramersV/HSCChemistry/Year2Band                                                |               0.135   |  0.13489     | -0.000109517 | True               |
| CramersV/HSCChemistry/Trajectory                                               |               0.081   |  0.0806647   | -0.000335328 | True               |
| CramersV/HSCChemistry/research                                                 |               0.107   |  0.106547    | -0.000452818 | True               |
| CramersV/CurrentInstitution/Year1Band                                          |               0.169   |  0.169216    |  0.000215563 | True               |
| CramersV/CurrentInstitution/Year2Band                                          |               0.133   |  0.132841    | -0.000159389 | True               |
| CramersV/CurrentInstitution/Trajectory                                         |               0.125   |  0.125114    |  0.000114108 | True               |
| CramersV/CurrentInstitution/research                                           |               0.327   |  0.32703     |  2.98872e-05 | True               |
| CramersV/CollegeCategory/Year1Band                                             |               0.176   |  0.17577     | -0.000229798 | True               |
| CramersV/CollegeCategory/Year2Band                                             |               0.128   |  0.128048    |  4.75013e-05 | True               |
| CramersV/CollegeCategory/Trajectory                                            |               0.08    |  0.0804507   |  0.000450709 | True               |
| CramersV/CollegeCategory/research                                              |               0.254   |  0.253547    | -0.000453294 | True               |
| CramersV/Gender/Year1Band                                                      |               0.118   |  0.117675    | -0.000324961 | True               |
| CramersV/Gender/Year2Band                                                      |               0.122   |  0.121922    | -7.75362e-05 | True               |
| CramersV/Gender/Trajectory                                                     |               0.188   |  0.187897    | -0.000103319 | True               |
| CramersV/Gender/research                                                       |               0.024   |  0.0240297   |  2.96942e-05 | True               |
| CramersV/LivingArrangement/Year1Band                                           |               0.104   |  0.104173    |  0.000173045 | True               |
| CramersV/LivingArrangement/Year2Band                                           |               0.079   |  0.0794197   |  0.000419674 | True               |
| CramersV/LivingArrangement/Trajectory                                          |               0.135   |  0.134542    | -0.000457982 | True               |
| CramersV/LivingArrangement/research                                            |               0.195   |  0.195322    |  0.00032161  | True               |
| CramersV/StudentClubMember/Year1Band                                           |               0.064   |  0.0641045   |  0.00010454  | True               |
| CramersV/StudentClubMember/Year2Band                                           |               0.053   |  0.053015    |  1.50329e-05 | True               |
| CramersV/StudentClubMember/Trajectory                                          |               0.087   |  0.0871055   |  0.000105506 | True               |
| CramersV/StudentClubMember/research                                            |               0.252   |  0.2522      |  0.000199624 | True               |
| CramersV/PartTimeEmployment/Year1Band                                          |               0.069   |  0.0686534   | -0.000346596 | True               |
| CramersV/PartTimeEmployment/Year2Band                                          |               0.049   |  0.0489921   | -7.91744e-06 | True               |
| CramersV/PartTimeEmployment/Trajectory                                         |               0.069   |  0.0686375   | -0.00036253  | True               |
| CramersV/PartTimeEmployment/research                                           |               0.153   |  0.15332     |  0.000320178 | True               |
| CramersV/ComputerAvailability/Year1Band                                        |               0.072   |  0.0724782   |  0.000478189 | True               |
| CramersV/ComputerAvailability/Year2Band                                        |               0.058   |  0.0581716   |  0.000171647 | True               |
| CramersV/ComputerAvailability/Trajectory                                       |               0.099   |  0.0986545   | -0.00034552  | True               |
| CramersV/ComputerAvailability/research                                         |               0.025   |  0.0247966   | -0.000203363 | True               |
| CramersV/MotherEducation/Year1Band                                             |               0.101   |  0.100621    | -0.000378751 | True               |
| CramersV/MotherEducation/Year2Band                                             |               0.09    |  0.0896404   | -0.000359567 | True               |
| CramersV/MotherEducation/Trajectory                                            |               0.084   |  0.0835451   | -0.000454924 | True               |
| CramersV/MotherEducation/research                                              |               0.099   |  0.0986956   | -0.00030437  | True               |
| CramersV/HouseholdIncomeYearly/Year1Band                                       |               0.078   |  0.0780391   |  3.90531e-05 | True               |
| CramersV/HouseholdIncomeYearly/Year2Band                                       |               0.052   |  0.0519074   | -9.263e-05   | True               |
| CramersV/HouseholdIncomeYearly/Trajectory                                      |               0.031   |  0.0310139   |  1.39241e-05 | True               |
| CramersV/HouseholdIncomeYearly/research                                        |               0.091   |  0.0905015   | -0.000498479 | True               |
| CramersV/DepartmentPreferenceOrder/Year1Band                                   |               0.065   |  0.0646416   | -0.000358393 | True               |
| CramersV/DepartmentPreferenceOrder/Year2Band                                   |               0.064   |  0.0637737   | -0.000226337 | True               |
| CramersV/DepartmentPreferenceOrder/Trajectory                                  |               0.089   |  0.0886622   | -0.00033783  | True               |
| CramersV/DepartmentPreferenceOrder/research                                    |               0.058   |  0.0581122   |  0.000112222 | True               |
| CramersV/UniversityPreferenceOrder/Year1Band                                   |               0.058   |  0.0579053   | -9.46792e-05 | True               |
| CramersV/UniversityPreferenceOrder/Year2Band                                   |               0.046   |  0.046458    |  0.000458031 | True               |
| CramersV/UniversityPreferenceOrder/Trajectory                                  |               0.076   |  0.0764348   |  0.000434827 | True               |
| CramersV/UniversityPreferenceOrder/research                                    |               0.089   |  0.0894399   |  0.000439949 | True               |
| CramersV/FinancialSupportSource/Year1Band                                      |               0.106   |  0.106       | -3.0624e-07  | True               |
| CramersV/FinancialSupportSource/Year2Band                                      |               0.085   |  0.0854018   |  0.000401767 | True               |
| CramersV/FinancialSupportSource/Trajectory                                     |               0.109   |  0.108671    | -0.000329482 | True               |
| CramersV/FinancialSupportSource/research                                       |               0.052   |  0.0522915   |  0.000291535 | True               |
| CramersV/CollegeLocation/Year1Band                                             |               0.088   |  0.087876    | -0.000123989 | True               |
| CramersV/CollegeLocation/Year2Band                                             |               0.07    |  0.070456    |  0.000455969 | True               |
| CramersV/CollegeLocation/Trajectory                                            |               0.017   |  0.0167789   | -0.000221105 | True               |
| CramersV/CollegeLocation/research                                              |               0.148   |  0.147732    | -0.000268222 | True               |
| CramersV/FatherEducation/Year1Band                                             |               0.086   |  0.0857683   | -0.000231671 | True               |
| CramersV/FatherEducation/Year2Band                                             |               0.063   |  0.0628639   | -0.00013605  | True               |
| CramersV/FatherEducation/Trajectory                                            |               0.088   |  0.0875254   | -0.000474568 | True               |
| CramersV/FatherEducation/research                                              |               0.103   |  0.103347    |  0.000347075 | True               |
| adjusted/year2/Year1Band                                                       |              11.1     | 11.3469      |  0.246943    | False              |
| adjusted/year1/HSCMathOrdinal                                                  |               1.7     |  1.78394     |  0.0839416   | False              |
| adjusted/year1/HSCPhysicsOrdinal                                               |               1.24    |  1.26091     |  0.0209065   | False              |
| adjusted/decline/HSCChemistryOrdinal                                           |               0.75    |  0.771943    |  0.0219433   | False              |
| adjusted/decline/C(Gender, Treatment(reference='Female'))[T.Male]              |               1.82    |  1.65461     | -0.165393    | False              |
| adjusted/improvement/C(Gender, Treatment(reference='Female'))[T.Male]          |               2.24    |  2.14643     | -0.0935668   | False              |
| adjusted/decline/Year1Band                                                     |               1.65    |  1.64349     | -0.00650937  | True               |
| adjusted/improvement/Year1Band                                                 |               0.6     |  0.587892    | -0.0121078   | False              |
| adjusted/research/StudentClubMemberOrdinal                                     |               1.354   |  1.36129     |  0.00729015  | True               |
| adjusted/research/C(CollegeCategory, Treatment(reference='Private'))[T.Public] |               0.669   |  0.680648    |  0.0116482   | False              |
| interaction/gender_HSCMath_year1                                               |               1e-12   |  3.10301e-12 |  2.10301e-12 | False              |
| interaction/gender_HSCPhysics_year1                                            |               1.4e-10 |  1.80381e-10 |  4.0381e-11  | False              |
| interaction/gender_HSCChemistry_year1                                          |               0.0003  |  0.000435558 |  0.000135558 | False              |
| interaction/gender_HSCMath_year2_without_year1                                 |               2.4e-12 |  3.50774e-10 |  3.48374e-10 | False              |
| interaction/gender_HSCMath_year2_with_year1                                    |               0.194   |  0.218076    |  0.0240758   | True               |
| interaction/clubs_college_research                                             |               0.034   |  0.0242397   | -0.00976028  | False              |
| interaction/clubs_faculty_research                                             |               0.0017  |  0.196159    |  0.194459    | False              |
| interaction/gender_living_decline                                              |               0.078   |  0.124935    |  0.0469352   | False              |
| interaction/trajectory_research_adjusted                                       |               0.0155  |  0.0124084   | -0.0030916   | True               |
| raw_rate/HSCMath/high_year1/190–200                                            |              58.1     | 58.1236      |  0.0235698   | True               |
| raw_rate/HSCMath/high_year1/All lower categories                               |              34.3     | 34.2544      | -0.0456472   | True               |
| raw_rate/HSCMath/high_year2/190–200                                            |              57.6     | 57.5515      | -0.0485126   | True               |
| raw_rate/HSCMath/high_year2/All lower categories                               |              32.6     | 32.6268      |  0.0267979   | True               |
| raw_rate/HSCPhysics/high_year1/190–200                                         |              59.8     | 59.8374      |  0.0373984   | True               |
| raw_rate/HSCPhysics/high_year1/All lower categories                            |              36       | 36.0221      |  0.0220614   | True               |
| raw_rate/HSCPhysics/high_year2/190–200                                         |              58       | 58.0488      |  0.0487805   | True               |
| raw_rate/HSCPhysics/high_year2/All lower categories                            |              34.7     | 34.7466      |  0.0466391   | True               |
| raw_rate/HSCChemistry/high_year1/190–200                                       |              55.2     | 55.2083      |  0.00833333  | True               |
| raw_rate/HSCChemistry/high_year1/All lower categories                          |              37.8     | 37.8129      |  0.0129117   | True               |
| raw_rate/HSCChemistry/high_year2/190–200                                       |              56.5     | 56.4583      | -0.0416667   | True               |
| raw_rate/HSCChemistry/high_year2/All lower categories                          |              36       | 36.0343      |  0.0342556   | True               |
| raw_rate/gender/Female/Improved                                                |               5.5     |  5.51948     |  0.0194805   | True               |
| raw_rate/gender/Female/Stable                                                  |              86       | 86.039       |  0.038961    | True               |
| raw_rate/gender/Female/Declined                                                |               8.4     |  8.44156     |  0.0415584   | True               |
| raw_rate/gender/Male/Improved                                                  |              13.7     | 13.704       |  0.00402802  | True               |
| raw_rate/gender/Male/Stable                                                    |              69.1     | 69.0455      | -0.0544658   | True               |
| raw_rate/gender/Male/Declined                                                  |              17.3     | 17.2504      | -0.0495622   | True               |
| raw_rate/clubs/No                                                              |              24.4     | 24.3583      | -0.0417258   | True               |
| raw_rate/clubs/Yes, one club/organization                                      |              45.9     | 45.9016      |  0.00163934  | True               |
| raw_rate/clubs/Yes, two clubs/organizations                                    |              46.9     | 46.875       | -0.025       | True               |
| raw_rate/clubs/Yes, three or more clubs/organizations                          |              54.8     | 54.8105      |  0.0104956   | True               |
| adjusted_probability/clubs/0                                                   |              29.9     | 30.2499      |  0.349871    | False              |
| adjusted_probability/clubs/1                                                   |              36       | 36.2994      |  0.299413    | False              |
| adjusted_probability/clubs/2                                                   |              42.5     | 42.7938      |  0.29377     | False              |
| adjusted_probability/clubs/3                                                   |              49.2     | 49.5494      |  0.349353    | False              |
| raw_rate/college/Public/high_year2                                             |              38.9     | 38.8654      | -0.0345523   | True               |
| raw_rate/college/Private/high_year2                                            |              38.8     | 38.7563      | -0.0436732   | True               |
| raw_rate/college/Public/low_year2                                              |              11.5     | 11.4862      | -0.0138303   | True               |
| raw_rate/college/Private/low_year2                                             |              19.1     | 19.0889      | -0.0110629   | True               |
| raw_rate/college/Public/research                                               |              25.8     | 25.7835      | -0.0165242   | True               |
| raw_rate/college/Private/research                                              |              50.6     | 50.6191      |  0.0190823   | True               |
| raw_rate/department_preference/1st choice                                      |              44.1     | 44.131       |  0.0310282   | True               |
| raw_rate/department_preference/5th choice or lower                             |              33.1     | 33.0579      | -0.0421488   | True               |
| raw_rate/trajectory_research/Declined                                          |              31       | 31.4516      |  0.451613    | True               |
| raw_rate/trajectory_research/Stable                                            |              36       | 35.7252      | -0.274751    | True               |
| raw_rate/trajectory_research/Improved                                          |              40       | 40.0531      |  0.0530504   | True               |
| faculty_rho/CSE                                                                |               0.843   |  0.843412    |  0.000412369 | True               |
| faculty_rho/EEE                                                                |               0.858   |  0.857681    | -0.000318912 | True               |
| faculty_rho/ME                                                                 |               0.821   |  0.820652    | -0.000348262 | True               |
| faculty_rho/CE                                                                 |               0.824   |  0.824145    |  0.000144608 | True               |
| faculty_rho/Others                                                             |               0.884   |  0.883555    | -0.000445472 | True               |
| adjusted_CI/year2/Year1Band/low                                                |               9.44    |  9.61988     |  0.179885    | False              |
| adjusted_CI/year2/Year1Band/high                                               |              13.04    | 13.3841      |  0.34406     | False              |
| adjusted_CI/year1/HSCMathOrdinal/low                                           |               1.49    |  1.55101     |  0.0610149   | False              |
| adjusted_CI/year1/HSCMathOrdinal/high                                          |               1.95    |  2.05185     |  0.101848    | False              |
| adjusted_CI/year1/HSCPhysicsOrdinal/low                                        |               1.08    |  1.09741     |  0.0174127   | False              |
| adjusted_CI/year1/HSCPhysicsOrdinal/high                                       |               1.42    |  1.44876     |  0.0287578   | False              |
| adjusted_CI/research/StudentClubMemberOrdinal/low                              |               1.25    |  1.25796     |  0.00795849  | True               |
| adjusted_CI/research/StudentClubMemberOrdinal/high                             |               1.46    |  1.47311     |  0.0131097   | False              |
