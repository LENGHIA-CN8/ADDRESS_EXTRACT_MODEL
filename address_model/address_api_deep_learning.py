import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
import yaml
import logging
from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import telegram
import torch

logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

logger.info('Loading model ...')
model = AutoModelForSeq2SeqLM.from_pretrained(config['model'], torch_dtype=torch.bfloat16, device_map="auto")
model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

# bot = telegram.Bot(token=config['BOT_TOKEN'])

# async def send_message_to_telegram(input_message: str, output_message: str):
#     full_message = f"Input: {input_message}\nOutput: {output_message}"
#     await bot.send_message(chat_id=config['CHAT_ID'], text=full_message)

app = FastAPI()

class Textmessage(BaseModel):
    message: str

@app.get("/")
def main():
    return {"message": "Welcome!"}

@app.post("/extract_address")
async def predict_name(background_tasks: BackgroundTasks, text_message: Textmessage):
    input_message = text_message.message
    logger.info(f'input message: {input_message}')
    b = tokenizer(input_message, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
              input_ids=b['input_ids'].to('cuda'),
              max_length=256,
              attention_mask=b['attention_mask'].to('cuda'),
          )
    message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f'output message: {message}')
    # background_tasks.add_task(send_message_to_telegram, input_message, message)
    if message == 'None':
        return {"Name": None}
    return {"Name": message}

if __name__ == "__main__":
    uvicorn.run(app, host="172.26.33.174", port=8006)