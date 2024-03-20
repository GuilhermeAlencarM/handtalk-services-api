
from fastapi.responses import JSONResponse
from http import HTTPStatus
from fastapi import APIRouter
from models.requests import Request
import joblib


detector = APIRouter()

@detector.post('/detect/sentiment')
async def dectect_sentiment(request: Request):
    """Detects the sentiment of a given text"""
    try:
        content = request.text
        naive_bays = joblib.load('./detector-model/modelo_sentimentos.joblib')
        prediction = naive_bays.predict([content])
        return JSONResponse(content={'status': HTTPStatus.ACCEPTED, 'sentiment': prediction[0]})
    except Exception as e:
        return JSONResponse(content={'status': HTTPStatus.INTERNAL_SERVER_ERROR, "content": f'{e}' })
    
@detector.post('/detect/language')
async def detect_language(request: Request):
    """
    Detects the language of a given text

    languages:
    -----------
    Estonian: Estoniano
    Swedish: Sueco
    English: Inglês
    Russian: Russo
    Romanian: Romeno
    Persian: Persa
    Pushto: Pashto
    Spanish: Espanhol
    Hindi: Hindi
    Korean: Coreano
    French: Francês
    Portuguese: Português
    Indonesian: Indonésio
    Urdu: Urdu
    Latin: Latim
    Turkish: Turco
    Dutch: Holandês
    Tamil: Tâmil
    Thai: Tailandês
    Arabic: Árabe
    -----------    
    """
    try:
        content = request.text
        naive_bays = joblib.load('./detector-model/modelo_idioma.joblib')
        prediction = naive_bays.predict([content])
        return JSONResponse(content={'status': HTTPStatus.ACCEPTED, 'language': prediction[0]})
    except Exception as e:
        return JSONResponse(content={'status': HTTPStatus.INTERNAL_SERVER_ERROR, "content": f'{e}' })