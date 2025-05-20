import { formatStintDate } from "@/app/utils";
import {
  Breadcrumbs,
  Container,
  Link,
  List,
  ListItem,
  Typography,
} from "@mui/material";
import { getAll } from "@vercel/edge-config";

export default async function Page({
  params,
}: {
  params: Promise<{ company: string }>;
}) {
  const { company } = await params;
  const professional_experiences = await getAll<ExperiencesResponse>([
    "professional_experiences",
  ]);
  const experience = professional_experiences.professional_experiences.find(
    (exp) => exp.slug === company
  );
  if (!experience) return <div>404</div>;
  return (
    <div>
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Home
        </Link>
        <Typography>{experience.company.name}</Typography>
      </Breadcrumbs>
      <Typography variant="h1">{experience.company.name}</Typography>
      {experience.stints.map((stint, index) => (
        <Container key={stint.title + index}>
          <Typography variant="h3">{stint.title}</Typography>
          <Typography variant="overline" sx={{ fontSize: "medium" }}>
            {formatStintDate(stint)}
          </Typography>
          <List>
            {stint.accomplishments.map(
              (accomplishment, accomplishmentIndex) => (
                <ListItem key={stint.title + accomplishmentIndex}>
                  {accomplishment}
                </ListItem>
              )
            )}
          </List>
        </Container>
      ))}
    </div>
  );
}
