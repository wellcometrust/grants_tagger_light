from loguru import logger


def parse_tags(tags):
    tags_list = []
    if tags is not None:
        filter_tags_list = list(filter(lambda x: x.strip() != "", tags.split(",")))
        tags_list = [str(y) for y in filter_tags_list]
        logger.info(f"Tags to be considered: {tags_list}")
    return tags_list


def parse_years(years):
    years_list = []
    if years is not None:
        filter_years_list = list(filter(lambda x: x.strip() != "", years.split(",")))
        years_list = [str(y) for y in filter_years_list]
        logger.info(f"Years to be considered: {years_list}")
    return years_list
