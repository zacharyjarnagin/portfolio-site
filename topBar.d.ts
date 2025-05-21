type Page = {
    label: string;
    route: string;
    icon?: OverridableComponent<SvgIconTypeMap<object, "svg">>
}

type Pages = Page[];