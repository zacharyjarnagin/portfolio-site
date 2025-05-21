type Page = {
    label: string;
    route: string;
    icon?: OverridableComponent<SvgIconTypeMap<{}, "svg">>
}

type Pages = Page[];