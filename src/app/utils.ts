const dateOptions: Intl.DateTimeFormatOptions = {
  month: "short",
  year: "numeric",
  timeZone: "UTC",
};

export const formatDateStr = (
  stintDate: string | undefined,
  includeDay: boolean = false
) => {
  if (!stintDate) return "Current";
  return new Date(Date.parse(stintDate)).toLocaleString("en-us", {
    ...dateOptions,
    ...(includeDay && { day: "numeric" }),
  });
};

export const formatStintDate = (stint: Stint) => {
  return (
    formatDateStr(stint.start_date) + " - " + formatDateStr(stint.end_date)
  );
};
