"""Dry run on a few bot-testing-area questions; nothing is published."""
import asyncio, sys, time
import dotenv; dotenv.load_dotenv(".env")
from forecasting_tools import MetaculusClient
import main as m

ids = [int(x) for x in sys.argv[1:]]
client = MetaculusClient()
qs = [client.get_question_by_post_id(i) for i in ids]
bot = m.FallTemplateBot2026(
    research_reports_per_question=1, predictions_per_research_report=5,
    use_research_summary_to_forecast=False, publish_reports_to_metaculus=False,
    folder_to_save_reports_to="dry_reports", skip_previously_forecasted_questions=False,
    extra_metadata_in_explanation=True,
    llms={
        "default": m.ThrottledLlm(model="gemini/gemma-4-26b-a4b-it", temperature=0.3, timeout=180, allowed_tries=3),
        "summarizer": m.ThrottledLlm(model="gemini/gemma-4-26b-a4b-it", timeout=120, allowed_tries=3),
        "researcher": "google-news",
        "parser": m.ThrottledLlm(model="gemini/gemma-4-26b-a4b-it", timeout=120, allowed_tries=3),
    },
)
t = time.time()
reps = asyncio.run(bot.forecast_questions(qs, return_exceptions=True))
for q, r in zip(qs, reps):
    print("=" * 80); print(q.page_url, type(q).__name__, q.question_text[:100])
    if isinstance(r, Exception): print("ERROR", type(r).__name__, str(r)[:500]); continue
    print("PREDICTION:", str(r.prediction)[:400])
print("elapsed %.0fs" % (time.time() - t))
