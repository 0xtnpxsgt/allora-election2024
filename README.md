
# Disclaimer...
**`This campaign rewards users who run worker nodes providing inferences for the US presidential election party winner once a day. Every inference should be the likelihood of the republican party winning the election. source`** [run-inference-political](https://app.allora.network/points/campaign/run-inference-political)

## 1. Components
- **Worker**: The node that publishes inferences to the Allora chain.
- **Inference**: A container that conducts inferences, maintains the model state, and responds to internal inference requests via a Flask application. This node operates with a basic linear regression model for price predictions.
- **Updater**: A cron-like container designed to update the inference node's data by daily fetching the latest market information from the data provider, ensuring the model stays current with new market trends.
- **Topic ID**: Running this worker on Topic 11
- **TOKEN= D** For have inference D: Democrat
- **TOKEN= R** For have inference R: Republic
- **MODEL**: 9 Total Models
- **Dataset**: polymarket.com
- **Probability**: Prediction of `%` total `0 - 100%`

### Setup Worker

1. **Clone this repository**
```bash
git clone https://github.com/0xtnpxsgt/allora-election2024.git
cd allora-election2024
```
2. **Provided and modify model config environment file**
    
Copy and read the example .env.example for your variables
```bash
cp .env.example .env
```
Edit .ENV Configuration
```bash
nano .env
 ```
	
3. **Here are the currently accepted configurations
- TOKEN= (`D` or `R`) 
- MODEL= Choose 1 Model from the LIST / Save `ctrl X + Y and Enter`
	- SVR
 	- RandomForest
	- GradientBoosting
	- LinearRegression
	- DecisionTree
	- KNeighbors
	- MLP
	- ExtraTrees
	- AdaBoost

- Sample .env Format
```bash
TOKEN=R
MODEL=MLP
```

4. **Edit your config & initialize worker**

Edit for WALLET NAME / SEEDPHRASE / RPC
```bash
nano config.json
```
Run the following commands root directory to initialize the worker
```bash
chmod +x init.config
./init.config
```
5. **Start the Services**
    
Run the following command to start the worker node, inference, and updater nodes:
```bash
docker compose up --build -d
```
Check logs
```bash
docker compose logs -f --tail=100
```

To confirm that the worker successfully sends the inferences to the chain, look for the following log:
```bash
{"level":"debug","msg":"Send Worker Data to chain","txHash":<tx-hash>,"time":<timestamp>,"message":"Success"}
```

## Testing Inference Only

Send requests to the inference model. For example, request probability of Democrat(`D`) or Republic(`R`) :
```sh
curl http://127.0.0.1:8000/inference/R
```
Expected response of numbering:
`"value":"xx.xxxx"`


###### THANKS TO ARCXTEAM FOR THE IDEA
