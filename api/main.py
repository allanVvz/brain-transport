from fastapi import FastAPI
from middleware.auth import auth_middleware
from routes import (evolution_webhook, health, messages, messaging_campaigns,
                    process, whatsapp)

app = FastAPI(title="Brain Transport", version="1.0.0")
app.middleware("http")(auth_middleware)
for router in (health.router, evolution_webhook.router, process.router,
               whatsapp.router, whatsapp.internal_router, messages.router,
               messaging_campaigns.router):
    app.include_router(router)
