"""FastAPI application for the healthcare assistant and its Streamlit UI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from app.api.routes.chat import router as chat_router
from app.api.routes.conversation import router as conversation_router
from app.api.routes.health import router as health_router
from app.database.db import Base, engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Healthcare Knowledge Assistant API",
    description="API used by the Streamlit healthcare assistant interface.",
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(conversation_router)