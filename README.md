Please note that:
- NLQ annotations have a known issue where ~14% of annotations have a near-0 query window and will result in under reported performance for the challenge (which will be corrected with a future dataset update): [NLQ Forum Post](https://discuss.ego4d-data.org/t/nlq-annotation-zero-temporal-windows/36)

# Ego4D Episodic Memory Benchmark

[EGO4D](https://ego4d-data.org/docs/) is the world's largest egocentric (first person) video ML dataset and benchmark suite.

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

The [Episodic Memory Benchmark](https://ego4d-data.org/docs/benchmarks/episodic-memory/) aims to make past video queryable and requires localizing where the answer can be seen within the user’s past video.  The repository contains the code needed to reproduce the results in the [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058).

There are 4 related tasks within a benchmark. Please see the README within each benchmark for details on setting up the codebase.

# [NLQ](./NLQ/README.md): *Natural Language Queries*

This task asks, "What/when/where....?" -- general natural language questions about the video past.    Given a video clip and a query expressed in natural language, the goal is to localize the temporal window within all the video history where the answer to the question is evident.  The task is novel because it requires searching through video to answer flexible linguistic queries.  For brevity, these example clips illustrate the video surrounding the ground truth (whereas the original input videos are each ~8 min). 

License

Ego4D is released under the MIT License.
