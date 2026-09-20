# WEKA research runtime

`weka.jar` is the unmodified WEKA 3.8.7 engine distributed with the local WEKA installation. It is included to rerun the existing nominal-attribute research-participation experiment without changing its implementation. Its GPL license is preserved in [COPYING](COPYING). This third-party license does not assign ownership or licensing terms to the project's datasets or original code.

Install a Java runtime (Java 8 or later; validated runtime recorded in the run manifest). Ensure `java` works on PATH, or set the `JAVA_EXE` environment variable to its executable. The Python entry point invokes Java in headless mode and captures the executed output. The JAR hash is recorded in the research metric JSON and artifact manifests.

Upstream: [WEKA, University of Waikato](https://www.cs.waikato.ac.nz/ml/weka/).
