from prefect import context

from cotune_llm.pipelines.helpers import cli, context_handler

if __name__ == "__main__":
    args = cli.parse_args()
    context_handler.initialize_context(args.config)
    context.pipeline()
