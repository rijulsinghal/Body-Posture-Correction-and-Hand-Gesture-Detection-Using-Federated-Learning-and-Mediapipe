import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self,rnd,results,failures):
        aggregated_weight = super().aggregate_fit(rnd,results,failures)
        # if aggregated_weight is not None:
            # print(f"Saving round{rnd} aggregated weights...")
            # np.savez(f"round-{rnd}-weights.",*aggregated_weight)
        return aggregated_weight

strategy = SaveModelStrategy()

fl.server.start_server(
    server_address = 'localhost:' + str(sys.argv[1]),
    config= {"num_rounds" : 1},
    grpc_max_message_length = 1024*1024*1024,
    strategy = strategy
)