const dateOptions: Intl.DateTimeFormatOptions = {
  month: "short",
  year: "numeric",
  timeZone: "UTC",
};

const formatDateStr = (stintDate: string | undefined) => {
  if (!stintDate) return "Current";
  return new Date(Date.parse(stintDate)).toLocaleString("en-gb", dateOptions);
}

export const formatStintDate = (stint: Stint) => {
    return formatDateStr(stint.start_date) + " - " + formatDateStr(stint.end_date)
};