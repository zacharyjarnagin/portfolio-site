import { Container, Divider, Typography } from "@mui/material";
import ExperienceTimeline from "./components/experienceTimeline";

export default async function Home() {
  return (
    <Container maxWidth="md">
      <Typography variant="h1" textAlign="center" fontSize={{ xs: 72, md: 96 }}>
        Zachary Jarnagin
      </Typography>
      <Container maxWidth="md" sx={{ marginTop: 2, marginBottom: 3 }}>
        <Typography textAlign="center">
          Hello! I am a software engineer in Boston, MA. I graduated from
          Northeastern University in &apos;22 with a Bachelor&apos;s of Science
          in computer science. I ran track & field and fell in love with
          computer science. When I am not coding 🧑‍💻, I&apos;m playing video
          games 🎮, watching F1 🏎️, cooking 🧑‍🍳, skiing 🎿, or playing with my
          dog 🐶.
        </Typography>
      </Container>
      <Divider />
      <Typography
        variant="h2"
        textAlign="center"
        fontSize={{ xs: 56, md: 72 }}
        marginBottom={2}
        marginTop={2}
      >
        My Experience
      </Typography>
      <ExperienceTimeline />
    </Container>
  );
}
