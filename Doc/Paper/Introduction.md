# Introduction

Internet of vehicles (IoV) is a paradigm that enables smart vehicles to communicate with each other and with roadside infrastructure, providing various services such as safety, entertainment, and navigation. However, IoV applications often require high computational power and low latency, which may not be satisfied by the limited on-board resources of smart vehicles. Mobile edge computing (MEC) is a promising technology that can augment the computing capabilities of smart vehicles by offloading their tasks to nearby edge servers. However, MEC faces several challenges in IoV scenarios, such as dynamic topology, heterogeneous resources, and diverse user preferences.

In this paper, we propose a reverse auction mechanism for IoV workflow scheduling in MEC with deep reinforcement learning (DRL). The basic idea is that the Edge Users(EUs) acts as an auctioneer who invites bids from Edge Servers(ESs) who are willing to provide computing services for IoV workflows. Then the Edge User determines which Edge Server to offload and the corresponding payment based on Edge Servers' bids and the user's preference. To cope with the uncertainty and complexity of IoV environment, we employ DRL techniques to learn the optimal bidding strategy for edge server owners. We evaluate our proposed mechanism through extensive simulations and compare it with several baseline methods. The results show that our mechanism can achieve better performance in terms of social welfare, user satisfaction, resource utilization, and incentive compatibility than existing methods.

