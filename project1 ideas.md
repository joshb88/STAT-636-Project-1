## 1. Twitch social network — does network position relate to streamer success?

We can use the one of the two SNAP twitch gamer data sets ([Stanford SNAP — Twitch Gamers Social Network](https://snap.stanford.edu/data/twitch_gamers.html) or the smaller alternative [Stanford SNAP — Twitch Social Networks](https://snap.stanford.edu/data/twitch-social-networks.html))

The idea would be to use one of the dataset's to look at whether a streamers positions in the Twitch social network is related to measures of success on the platform.

The main question could be:

**Are streamers who occupy more prominent or well-connected positions in the Twitch social network more likely to be successful?**

Some other auxilliary questions:

- Is a streamer’s number of connections associated with their view count?
- Are users with higher network centrality more likely to have Twitch affiliate status?
- Does being part of a densely connected portion of the network relate to streamer success?
- Do network measures such as degree, clustering coefficient, PageRank/eigenvector centrality, or betweenness centrality provide useful information for predicting view count or affiliate status?
- Does network position still matter after accounting for other available user characteristics?

I like the diea of staying simple for the first project.

## 2. GitHub social network — do developer specialties relate to network structure?

Another project idea that's sort of similar to the first. Using the [GitHub Social Network dataset](https://snap.stanford.edu/data/github-social.html), determine whether web developers and machine learning developers occupy different positions in the GitHub social network.

The nodes are GitHub developers and the edges are mutual follower relationships, andeach developer also has node features based on information such as repositories starred, location, employer, and email information. The dataset includes a binary label identifying users as either web developers or machine learning developers, based on their job titles.

The main question could be:

**Do web developers and machine-learning developers have measurably different positions or connection patterns within the GitHub social network?**

Some more specific questions could be:

- Do machine-learning and web developers differ in their average number of connections?
- Do the two groups differ in clustering, centrality, or other network characteristics?
- Do developers of the same type tend to connect to one another more often than would be expected by chance?
- Can a developer’s network characteristics help predict whether they are a web developer or a machine-learning developer?
- Does adding network information improve classification compared with using the supplied user attributes alone?

One possible structure would be to compare three relatively simple models:

1. a model using only the supplied user attributes,
2. a model using only network-derived characteristics such as degree, clustering, and centrality,
3. a model using both sets of variables.

That would let us ask not only whether the groups differ, but whether **the structure of a developer’s social network contains useful information beyond their ordinary profile characteristics**.

This dataset is smaller than the full Twitch dataset and is already set up specifically for node classification, so it seems like it might be similarly simple/approachable for a first project.
