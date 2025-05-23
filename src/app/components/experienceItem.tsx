"use client";
import * as React from "react";
import Typography from "@mui/material/Typography";
import { Avatar, Grid } from "@mui/material";
import { formatStintDate } from "../utils";
import theme from "@/theme";

const AVATAR_SIZE = 5;

export default function ExperienceItem({
  experience,
}: {
  experience: Experience;
}) {
  return (
    <Grid alignItems={"center"} paddingBottom={2} container tabIndex={0}>
      <Grid
        container
        size={12}
        direction="row"
        sx={{
          justifyContent: { xs: "center", md: "flex-start" },
          alignItems: "center",
          gap: 2,
        }}
      >
        {experience.company.iconUri && (
          <Avatar
            src={experience.company.iconUri}
            sx={{
              boxShadow: 8,
              width: theme.spacing(AVATAR_SIZE),
              height: theme.spacing(AVATAR_SIZE),
              padding: 0.25,
              backgroundColor: "#fff",
            }}
          />
        )}
        <Typography variant="h4" marginLeft={1}>
          {experience.company.name}
        </Typography>
      </Grid>
      {experience.stints.map((stint, index) => (
        <Grid
          key={index}
          size={12}
          sx={{ marginTop: 2, marginLeft: { md: 10 } }}
        >
          <Typography variant="h5">{stint.title}</Typography>
          <Typography variant="overline">{formatStintDate(stint)}</Typography>
          {stint.accomplishments.map((accomplishment, accomplishmentIndex) => (
            <Typography
              key={accomplishmentIndex}
              sx={{ marginLeft: 2 }}
              component="li"
            >
              {accomplishment}
            </Typography>
          ))}
        </Grid>
      ))}
    </Grid>
  );
}
