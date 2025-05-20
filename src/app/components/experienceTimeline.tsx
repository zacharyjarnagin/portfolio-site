import * as React from "react";
import { getAll } from "@vercel/edge-config";
import { Box, CircularProgress, Divider } from "@mui/material";
import ExperienceItem from "./experienceItem";

export default async function ExperienceTimeline() {
  const professional_experiences = await getAll<ExperiencesResponse>([
    "professional_experiences",
  ]);
  if (
    !professional_experiences ||
    !professional_experiences.professional_experiences
  )
    return <CircularProgress />;
  return (
    <Box>
      {professional_experiences?.professional_experiences?.map(
        (experience, index) => (
          <Box key={experience.company.name} sx={{ marginBottom: 2 }}>
            <ExperienceItem experience={experience} />
            {index <
              professional_experiences.professional_experiences.length - 1 && (
              <Divider />
            )}
          </Box>
        )
      )}
    </Box>
  );
}
